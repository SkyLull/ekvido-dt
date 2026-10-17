/*
    This file is part of darktable (ekvido fork),
    copyright (c) 2026 Jack Chen.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/* ekvido mist: physically-based diffusion (white/black mist) whose glow scales with true
   scene intensity. Processing lives in common/ekvido/ (pure C, tested against the Python
   reference); this file is the darktable glue. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "bauhaus/bauhaus.h"
#include "common/ekvido/kernel_view.h"
#include "common/ekvido/mist.h"
#include "common/ekvido/psf.h"
#include "common/ekvido/roi.h"
#ifdef HAVE_OPENCL
#include "common/ekvido/cl/ekv_cl_dt.h"
#include "common/opencl.h"
#include "develop/tiling.h"
#endif
#include "common/imagebuf.h"
#include "control/control.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "dtgtk/drawingarea.h"
#include "dtgtk/paint.h"
#include "dtgtk/togglebutton.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

DT_MODULE_INTROSPECTION(1, dt_iop_ekvidomist_params_t)

typedef struct dt_iop_ekvidomist_params_t
{
  float strength;       // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.15 $DESCRIPTION: "scatter amount"
  float halo_radius;    // $MIN: 0.0005 $MAX: 0.05 $DEFAULT: 0.004 $DESCRIPTION: "halo size"
  float halo_beta;      // $MIN: 0.6 $MAX: 4.0 $DEFAULT: 1.5 $DESCRIPTION: "halo falloff"
  float halo_stretch_x; // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "halo width"
  float halo_stretch_y; // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "halo height"
  float veil_radius;    // $MIN: 0.02 $MAX: 0.25 $DEFAULT: 0.15 $DESCRIPTION: "haze size"
  float veil_beta;      // $MIN: 0.6 $MAX: 3.0 $DEFAULT: 1.0 $DESCRIPTION: "haze falloff"
  float w_halo;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "halo share"
  float w_veil;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "haze share"
  int arms;             // $MIN: 0 $MAX: 8 $DEFAULT: 0 $DESCRIPTION: "star arms"
  float arm_angle;      // $MIN: 0.0 $MAX: 180.0 $DEFAULT: 0.0 $DESCRIPTION: "star rotation"
  float arm_radius;     // $MIN: 0.0001 $MAX: 0.02 $DEFAULT: 0.001 $DESCRIPTION: "star core size"
  float arm_beta;       // $MIN: 0.6 $MAX: 4.0 $DEFAULT: 1.3 $DESCRIPTION: "star falloff"
  float arm_sharpness;  // $MIN: 0.2 $MAX: 2.0 $DEFAULT: 0.3 $DESCRIPTION: "star shape"
  float arm_stretch_x;  // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "star width"
  float arm_stretch_y;  // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "star height"
  float w_arms;         // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "star share"
} dt_iop_ekvidomist_params_t;

typedef struct dt_iop_ekvidomist_gui_data_t
{
  dt_gui_collapsible_section_t halo_section, haze_section, star_section;
  GtkDrawingArea *area;          // kernel preview pane
  unsigned char *view;           // cached BGRA kernel image, view_width^2
  int view_width;
  gboolean view_dirty;
  float view_diag;               // diagonal (px) the view was built at, for the scale bar
  GtkWidget *glow_toggle;
  gboolean show_glow;            // GUI-only: darkroom shows the glow alone
} dt_iop_ekvidomist_gui_data_t;

typedef struct dt_iop_ekvidomist_data_t
{
  dt_iop_ekvidomist_params_t p;
  ekvido_mist_cache_t *cache;
#ifdef HAVE_OPENCL
  ekv_cl_cache_t *clcache;
#endif
} dt_iop_ekvidomist_data_t;

const char *name()
{
  return _("ekvido mist");
}

const char *aliases()
{
  return _("mist|diffusion|pro-mist|bloom|glow");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("physically-based mist filter: glow that scales with true highlight intensity"),
                                _("creative"), _("linear, RGB, scene-referred"), _("linear, RGB"),
                                _("linear, RGB, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_EFFECTS | IOP_GROUP_EFFECT;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

static void _spec_from_params(const dt_iop_ekvidomist_params_t *p, ekvido_kernel_spec_t *s)
{
  *s = (ekvido_kernel_spec_t){ .halo_radius = p->halo_radius, .halo_beta = p->halo_beta,
                               .halo_stretch_x = p->halo_stretch_x, .halo_stretch_y = p->halo_stretch_y,
                               .veil_radius = p->veil_radius, .veil_beta = p->veil_beta,
                               .arms = p->arms, .arm_angle = p->arm_angle, .arm_radius = p->arm_radius,
                               .arm_beta = p->arm_beta, .arm_sharpness = p->arm_sharpness,
                               .arm_stretch_x = p->arm_stretch_x, .arm_stretch_y = p->arm_stretch_y,
                               .w_halo = p->w_halo, .w_veil = p->w_veil, .w_arms = p->w_arms };
}

/* Glow sizes are fractions of the module's full input diagonal, at this pipe's scale. */
static float _diag_px(const dt_dev_pixelpipe_iop_t *piece, const float scale)
{
  return hypotf((float)piece->buf_in.width, (float)piece->buf_in.height) * scale;
}

void modify_roi_in(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *const roi_out,
                   dt_iop_roi_t *roi_in)
{
  *roi_in = *roi_out;
  const dt_iop_ekvidomist_data_t *d = piece->data;
  ekvido_kernel_spec_t s;
  _spec_from_params(&d->p, &s);
  const dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  const gboolean need = d->p.strength > 0.f || (g && g->show_glow);
  const int margin = need ? ekvido_kernel_half_size(&s, _diag_px(piece, roi_out->scale)) : 0;
  const ekvido_rect_t r = ekvido_expand_rect(
      (ekvido_rect_t){ roi_out->x, roi_out->y, roi_out->width, roi_out->height }, margin,
      (int)floorf(piece->buf_in.width * roi_out->scale), (int)floorf(piece->buf_in.height * roi_out->scale));
  roi_in->x = r.x;
  roi_in->y = r.y;
  roi_in->width = r.w;
  roi_in->height = r.h;
}

/* mist's output is a crop of its input (zoom margin), so masks are cropped the same way;
   without this, raster masks used further down the pipe fail whenever mist adds a margin. */
void distort_mask(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const float *const in, float *const out,
                  const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_copy_image_roi(out, in, 1, roi_in, roi_out);
}

void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out)) return;
  dt_iop_ekvidomist_data_t *d = piece->data;
  ekvido_mist_params_t mp = { .strength = d->p.strength };
  _spec_from_params(&d->p, &mp.spec);
  const dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  const gboolean glow_only = g && g->show_glow && self->dev->gui_attached && dt_pipe_is_full(piece->pipe);
  const gboolean interactive = self->dev->gui_attached && !(piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT);
  const ekvido_rect_t rect = { roi_out->x - roi_in->x, roi_out->y - roi_in->y, roi_out->width, roi_out->height };
  const int err = ekvido_mist_process(ivoid, roi_in->width, roi_in->height, ovoid, rect, &mp,
                                      _diag_px(piece, roi_in->scale), interactive, glow_only,
                                      interactive ? d->cache : NULL, (int)dt_get_num_threads());
  if(err)
  {
    dt_iop_copy_image_roi(ovoid, ivoid, 4, roi_in, roi_out);
    dt_control_log(_("ekvido mist: not enough memory, module skipped"));
  }
}

#ifdef HAVE_OPENCL
/* GPU twin of process(); any failure falls back to process() (spec D8). */
int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4, self, piece->colors, (void *)dev_in, (void *)dev_out, roi_in, roi_out))
    return DT_OPENCL_PROCESS_CL;
  ekv_cl_t *E = ekv_cl_dt_get(piece->pipe->devid);
  if(!E) return DT_OPENCL_PROCESS_CL;
  dt_iop_ekvidomist_data_t *d = piece->data;
  ekvido_mist_params_t mp = { .strength = d->p.strength };
  _spec_from_params(&d->p, &mp.spec);
  const dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  const gboolean glow_only = g && g->show_glow && self->dev->gui_attached && dt_pipe_is_full(piece->pipe);
  const gboolean interactive = self->dev->gui_attached && !(piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT);
  const ekvido_rect_t rect = { roi_out->x - roi_in->x, roi_out->y - roi_in->y, roi_out->width, roi_out->height };
  const int err = ekv_cl_mist_process(E, dev_in, roi_in->width, roi_in->height, dev_out, rect, &mp,
                                      _diag_px(piece, roi_in->scale), interactive, glow_only,
                                      interactive ? d->clcache : NULL);
  return err ? DT_OPENCL_DEFAULT_ERROR : CL_SUCCESS;
}

void tiling_callback(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out, dt_develop_tiling_t *tiling)
{
  const dt_iop_ekvidomist_data_t *d = piece->data;
  ekvido_kernel_spec_t s;
  _spec_from_params(&d->p, &s);
  const float diag = _diag_px(piece, roi_in->scale);
  const gboolean interactive = self->dev->gui_attached && !(piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT);
  const int pad = interactive ? ekvido_kernel_half_cap(diag) : ekvido_kernel_half_size(&s, diag);
  const double px = (double)roi_in->width * roi_in->height;
  const double extra = (double)ekv_cl_mist_bytes(roi_in->width, roi_in->height, pad)
                       + (interactive ? 3.0 * sizeof(float) * px : 0.0);
  tiling->factor = 2.0f;
  tiling->factor_cl = (float)(2.0 + extra / (16.0 * fmax(px, 1.0)));
  tiling->maxbuf = 1.0f;
  tiling->maxbuf_cl = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = 0;
  tiling->align = 1;
}
#endif

void init_global(dt_iop_module_so_t *self)
{
#ifdef HAVE_OPENCL
  ekv_cl_dt_retain();
#endif
  self->data = NULL;
}

void cleanup_global(dt_iop_module_so_t *self)
{
#ifdef HAVE_OPENCL
  ekv_cl_dt_release();
#endif
  self->data = NULL;
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_ekvidomist_data_t *d = calloc(1, sizeof(dt_iop_ekvidomist_data_t));
  d->cache = ekvido_mist_cache_new();
#ifdef HAVE_OPENCL
  d->clcache = ekv_cl_cache_new();
#endif
  piece->data = d;
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_ekvidomist_data_t *d = piece->data;
  if(d)
  {
    ekvido_mist_cache_free(d->cache);
#ifdef HAVE_OPENCL
    ekv_cl_cache_free(d->clcache);
#endif
  }
  free(d);
  piece->data = NULL;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_ekvidomist_data_t *d = piece->data;
  memcpy(&d->p, p1, sizeof(d->p));
}

static dt_iop_ekvidomist_params_t _defaults(void)
{
  return (dt_iop_ekvidomist_params_t){ .strength = 0.15f, .halo_radius = 0.004f, .halo_beta = 1.5f,
                                       .halo_stretch_x = 1.f, .halo_stretch_y = 1.f, .veil_radius = 0.15f,
                                       .veil_beta = 1.f, .w_halo = 0.5f, .w_veil = 0.5f, .arms = 0,
                                       .arm_angle = 0.f, .arm_radius = 0.001f, .arm_beta = 1.3f,
                                       .arm_sharpness = 0.3f, .arm_stretch_x = 1.f, .arm_stretch_y = 1.f,
                                       .w_arms = 0.f };
}

void init_presets(dt_iop_module_so_t *self)
{
  dt_iop_ekvidomist_params_t p = _defaults();
  p.w_halo = 0.4f;
  p.w_veil = 0.6f;
  dt_gui_presets_add_generic(_("mist white"), self->op, self->version(), &p, sizeof(p), TRUE,
                             DEVELOP_BLEND_CS_RGB_SCENE);
  p = _defaults();
  p.w_halo = 0.85f;
  p.w_veil = 0.15f;
  p.halo_beta = 1.8f;
  dt_gui_presets_add_generic(_("mist black"), self->op, self->version(), &p, sizeof(p), TRUE,
                             DEVELOP_BLEND_CS_RGB_SCENE);
}

static GtkWidget *_slider(dt_iop_module_t *sect, const char *param, const char *tooltip, int digits)
{
  GtkWidget *w = dt_bauhaus_slider_from_params(sect, param);
  if(digits >= 0) dt_bauhaus_slider_set_digits(w, digits);
  gtk_widget_set_tooltip_text(w, tooltip);
  return w;
}

static dt_iop_module_t *_section(dt_iop_module_t *self, dt_iop_module_t *out, dt_gui_collapsible_section_t *cs,
                                 const char *conf, const char *label)
{
  dt_gui_new_collapsible_section(cs, conf, label, GTK_BOX(self->widget), DT_ACTION(self));
  *out = DT_IOP_SECTION_FOR_PARAMS_DECL(self, label, cs->container);
  return out;
}

static gboolean _draw_kernel(GtkWidget *widget, cairo_t *cr, dt_iop_module_t *self)
{
  dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  GtkAllocation a;
  gtk_widget_get_allocation(widget, &a);
  const int width = a.width;
  if(width < 16) return FALSE;
  if(width != g->view_width || g->view_dirty || !g->view)
  {
    free(g->view);
    g->view = malloc((size_t)width * width * 4);
    if(!g->view) { g->view_width = 0; return FALSE; }
    ekvido_kernel_spec_t s;
    _spec_from_params(self->params, &s);
    g->view_diag = ekvido_kernel_view(&s, width, g->view);
    g->view_width = width;
    g->view_dirty = FALSE;
  }
  const int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, width);
  if(stride != width * 4) return FALSE; /* ARGB32 is always 4 bytes/pixel; guard anyway */
  cairo_surface_t *surf = cairo_image_surface_create_for_data(g->view, CAIRO_FORMAT_ARGB32, width, width, stride);
  cairo_set_source_surface(cr, surf, 0, 0);
  cairo_paint(cr);
  cairo_surface_destroy(surf);

  /* scale bar: the largest round fraction of the diagonal that fits in 40 % of the pane */
  static const double fracs[] = { 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005 };
  for(int i = 0; i < 9; i++)
  {
    const double len = fracs[i] * g->view_diag;
    if(len > 0.4 * width) continue;
    char label[32];
    snprintf(label, sizeof(label), "%g diag", fracs[i]);
    cairo_set_source_rgb(cr, 0.9, 0.9, 0.9);
    cairo_set_line_width(cr, 2.0);
    cairo_move_to(cr, 8, width - 10);
    cairo_line_to(cr, 8 + len, width - 10);
    cairo_stroke(cr);
    cairo_move_to(cr, 8, width - 16);
    cairo_show_text(cr, label);
    break;
  }
  return TRUE;
}

static void _set_show_glow(dt_iop_module_t *self, gboolean on)
{
  dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  if(g->show_glow == on) return;
  g->show_glow = on;
  ++darktable.gui->reset;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->glow_toggle), on);
  --darktable.gui->reset;
  dt_iop_refresh_center(self);
}

/* Connected to "button-press-event": returns TRUE so GTK's own toggle on release does not
   flip the button back — we set its state ourselves in _set_show_glow(). */
static gboolean _glow_toggled(GtkWidget *button, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return TRUE;
  dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  dt_iop_request_focus(self);
  if(self->request_mask_display)
  {
    dt_control_log(_("cannot show the glow while a blend mask is displayed"));
    _set_show_glow(self, FALSE);
    return TRUE;
  }
  const gboolean on = !g->show_glow;
  if(on)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE); /* enable the module */
    for(GList *l = self->dev->iop; l; l = g_list_next(l))   /* only one instance previews at a time */
    {
      dt_iop_module_t *m = l->data;
      if(m != self && m->gui_data && !strcmp(m->op, self->op)) _set_show_glow(m, FALSE);
    }
  }
  _set_show_glow(self, on);
  return TRUE;
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  if(!in && self->gui_data) _set_show_glow(self, FALSE);
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  g->view_dirty = TRUE;
  gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  free(g->view);
  g->view = NULL;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_ekvidomist_gui_data_t *g = IOP_GUI_ALLOC(ekvidomist);
  g->area = GTK_DRAWING_AREA(dtgtk_drawing_area_new_with_height(0));
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->area),
                              _("glow shape on a log scale; the bar shows a fraction of the image diagonal"));
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_draw_kernel), self);
  g->glow_toggle = dt_iop_togglebutton_new(self, NULL, N_("show glow only"), NULL, G_CALLBACK(_glow_toggled),
                                           FALSE, 0, 0, dtgtk_cairo_paint_showmask, NULL);
  gtk_widget_set_tooltip_text(g->glow_toggle, _("show only the glow in the centre view (not saved, not exported)"));
  g->view_dirty = TRUE;
  self->widget = dt_gui_vbox(GTK_WIDGET(g->area),
                             dt_gui_hbox(dt_gui_expand(dt_ui_label_new(_("show glow only"))), g->glow_toggle));

  _slider(self, "strength", _("fraction of all light the filter scatters (0.15 = 15 %)"), 3);

  dt_iop_module_t sec;
  dt_iop_module_t *s = _section(self, &sec, &g->halo_section, "plugins/darkroom/ekvidomist/expand_halo", N_("halo"));
  _slider(s, "halo_radius", _("size of the main soft glow (fraction of the image diagonal)"), 4);
  _slider(s, "halo_beta", _("low: long misty tail; high: tight glow"), 2);
  _slider(s, "halo_stretch_x", _("stretch the halo horizontally"), 2);
  _slider(s, "halo_stretch_y", _("stretch the halo vertically"), 2);
  _slider(s, "w_halo", _("share of the glow in the halo (black mist: high)"), 2);

  s = _section(self, &sec, &g->haze_section, "plugins/darkroom/ekvidomist/expand_haze", N_("haze"));
  _slider(s, "veil_radius", _("size of the very wide, faint haze (fraction of the image diagonal)"), 3);
  _slider(s, "veil_beta", _("low: haze reaches further"), 2);
  _slider(s, "w_veil", _("share of the glow in the wide haze (white mist: high)"), 2);

  s = _section(self, &sec, &g->star_section, "plugins/darkroom/ekvidomist/expand_star", N_("star"));
  _slider(s, "arms", _("number of arms; 4 = cross, 2 = horizontal streak, 0 = off"), -1);
  GtkWidget *angle = _slider(s, "arm_angle", _("angle of the first arm"), 1);
  dt_bauhaus_slider_set_format(angle, "°");
  _slider(s, "arm_radius", _("size of the star's bright centre (fraction of the image diagonal)"), 4);
  _slider(s, "arm_beta", _("low: long arms on bright sources; high: short arms"), 2);
  _slider(s, "arm_sharpness", _("2 round, 1 diamond, below 1 thinner arms"), 2);
  _slider(s, "arm_stretch_x", _("stretch the whole star horizontally"), 2);
  _slider(s, "arm_stretch_y", _("stretch the whole star vertically"), 2);
  _slider(s, "w_arms", _("share of the glow in the star (low: only the hottest points show arms)"), 2);
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_ekvidomist_gui_data_t *g = self->gui_data;
  dt_gui_update_collapsible_section(&g->halo_section);
  dt_gui_update_collapsible_section(&g->haze_section);
  dt_gui_update_collapsible_section(&g->star_section);
  g->view_dirty = TRUE;
  gtk_widget_queue_draw(GTK_WIDGET(g->area));
}
