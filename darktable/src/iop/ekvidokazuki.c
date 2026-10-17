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

/* ekvido kazuki: "Kazuki light" — highlights glow in their own warm or cool colour, with
   halo, haze and star shapes, laid on additively, energy-conserving, with Clip Studio
   Paint's layer modes, or as an overlay that leaves glow-free areas alone. Processing lives in common/ekvido/ (pure C, tested against the
   Python reference); this file is the darktable glue. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "bauhaus/bauhaus.h"
#include "common/ekvido/color.h"
#include "common/ekvido/energy.h"
#include "common/ekvido/kazuki.h"
#include "common/ekvido/kernel_view.h"
#include "common/ekvido/psf.h"
#include "common/ekvido/roi.h"
#ifdef HAVE_OPENCL
#include "common/ekvido/cl/ekv_cl_dt.h"
#include "common/opencl.h"
#include "develop/tiling.h"
#endif
#include "common/imagebuf.h"
#include "common/math.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/pixelpipe_hb.h"
#include "dtgtk/drawingarea.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

DT_MODULE_INTROSPECTION(2, dt_iop_ekvidokazuki_params_t)

/* Append only: the values are stored in edit histories (same order as ekvido_blend_mode_t). */
typedef enum dt_iop_ekvidokazuki_blend_t
{
  EKVIDOKAZUKI_BLEND_ADDITIVE = 0,    // $DESCRIPTION: "kazuki blend|additive"
  EKVIDOKAZUKI_BLEND_CONSERVE = 1,    // $DESCRIPTION: "kazuki blend|conserve"
  EKVIDOKAZUKI_BLEND_SCREEN = 2,      // $DESCRIPTION: "kazuki blend|screen"
  EKVIDOKAZUKI_BLEND_OVERLAY = 3,     // $DESCRIPTION: "kazuki blend|overlay"
  EKVIDOKAZUKI_BLEND_COLOR_DODGE = 4, // $DESCRIPTION: "kazuki blend|colour dodge"
  EKVIDOKAZUKI_BLEND_GLOW_DODGE = 5,  // $DESCRIPTION: "kazuki blend|glow dodge"
  EKVIDOKAZUKI_BLEND_OVERLAY_GLOW = 6, // $DESCRIPTION: "kazuki blend|overlay (glow only)"
} dt_iop_ekvidokazuki_blend_t;

typedef enum dt_iop_ekvidokazuki_clip_t
{
  EKVIDOKAZUKI_CLIP_AUTO = 0,   // $DESCRIPTION: "auto"
  EKVIDOKAZUKI_CLIP_MANUAL = 1, // $DESCRIPTION: "manual"
} dt_iop_ekvidokazuki_clip_t;

/* Version 2: version 1's 38 four-byte fields (152 bytes), then the source mask (spec §5.7):
   20 + 3 × 4 bytes, 184 in all, no padding. tools/dt_compare.py packs this layout
   (KAZUKI_FIELDS / "<fi3fif6f8f9fi7f20s3i") and its tests read the $DEFAULTs below. */
typedef struct dt_iop_ekvidokazuki_params_t
{
  float strength;                    // $MIN: 0.0 $MAX: 4.0 $DEFAULT: 0.5 $DESCRIPTION: "glow strength"
  dt_iop_ekvidokazuki_blend_t blend; // $DEFAULT: EKVIDOKAZUKI_BLEND_ADDITIVE $DESCRIPTION: "glow blend"
  float blend_white_ev;              // $MIN: -4.0 $MAX: 8.0 $DEFAULT: 0.0 $DESCRIPTION: "blend white (EV)"
  float threshold_ev;                // $MIN: -6.0 $MAX: 6.0 $DEFAULT: 0.0 $DESCRIPTION: "glow threshold (EV)"
  float knee_ev;                     // $MIN: 0.0 $MAX: 4.0 $DEFAULT: 1.0 $DESCRIPTION: "threshold softness (EV)"
  dt_iop_ekvidokazuki_clip_t clip_mode; // $DEFAULT: EKVIDOKAZUKI_CLIP_AUTO $DESCRIPTION: "clip detection"
  float clip_ev;                     // $MIN: -4.0 $MAX: 8.0 $DEFAULT: 0.0 $DESCRIPTION: "clip level (EV)"
  float hue_radius;                  // $MIN: 0.001 $MAX: 0.1 $DEFAULT: 0.01 $DESCRIPTION: "colour sampling area"
  float neutral_point;               // $MIN: -0.3 $MAX: 0.3 $DEFAULT: 0.0 $DESCRIPTION: "neutral point"
  float warmth_softness;             // $MIN: 0.005 $MAX: 0.3 $DEFAULT: 0.05 $DESCRIPTION: "warm/cool softness"
  float neutral_bias;                // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.8 $DESCRIPTION: "colourless source bias"
  float push_warm;                   // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.7 $DESCRIPTION: "warm colour pull"
  float push_cool;                   // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.7 $DESCRIPTION: "cool colour pull"
  float warm_core_hue;               // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 40.0 $DESCRIPTION: "warm centre hue"
  float warm_core_sat;               // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.45 $DESCRIPTION: "warm centre saturation"
  float warm_edge_hue;               // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 22.5 $DESCRIPTION: "warm edge hue"
  float warm_edge_sat;               // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.88 $DESCRIPTION: "warm edge saturation"
  float cool_core_hue;               // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 193.846154 $DESCRIPTION: "cool centre hue"
  float cool_core_sat;               // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.65 $DESCRIPTION: "cool centre saturation"
  float cool_edge_hue;               // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 232.173913 $DESCRIPTION: "cool edge hue"
  float cool_edge_sat;               // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.92 $DESCRIPTION: "cool edge saturation"
  float band_split;                  // $MIN: 0.0 $MAX: 0.05 $DEFAULT: 0.006 $DESCRIPTION: "centre/edge colour radius"
  float halo_radius;                 // $MIN: 0.0005 $MAX: 0.05 $DEFAULT: 0.006 $DESCRIPTION: "halo size"
  float halo_beta;                   // $MIN: 0.6 $MAX: 4.0 $DEFAULT: 1.3 $DESCRIPTION: "halo falloff"
  float halo_stretch_x;              // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 1.85 $DESCRIPTION: "halo width"
  float halo_stretch_y;              // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "halo height"
  float veil_radius;                 // $MIN: 0.02 $MAX: 0.25 $DEFAULT: 0.1 $DESCRIPTION: "haze size"
  float veil_beta;                   // $MIN: 0.6 $MAX: 3.0 $DEFAULT: 1.2 $DESCRIPTION: "haze falloff"
  float w_halo;                      // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 1.0 $DESCRIPTION: "halo share"
  float w_veil;                      // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "haze share"
  int arms;                          // $MIN: 0 $MAX: 8 $DEFAULT: 4 $DESCRIPTION: "star arms"
  float arm_angle;                   // $MIN: 0.0 $MAX: 180.0 $DEFAULT: 0.0 $DESCRIPTION: "star rotation"
  float arm_radius;                  // $MIN: 0.0001 $MAX: 0.02 $DEFAULT: 0.001 $DESCRIPTION: "star core size"
  float arm_beta;                    // $MIN: 0.6 $MAX: 4.0 $DEFAULT: 0.96 $DESCRIPTION: "star falloff"
  float arm_sharpness;               // $MIN: 0.2 $MAX: 2.0 $DEFAULT: 0.32 $DESCRIPTION: "star shape"
  float arm_stretch_x;               // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 2.93 $DESCRIPTION: "star width"
  float arm_stretch_y;               // $MIN: 0.25 $MAX: 5.0 $DEFAULT: 0.58 $DESCRIPTION: "star height"
  float w_arms;                      // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.2 $DESCRIPTION: "star share"
  /* v2: source mask = another module's raster mask, as blend params store one */
  char source_mask_op[20];           // op name of the module providing it; "" = none
  int source_mask_instance;          // $DEFAULT: 0 $DESCRIPTION: "source mask instance"
  int source_mask_id;                // $DEFAULT: 0 $DESCRIPTION: "source mask id"
  gboolean source_mask_invert;       // $DEFAULT: FALSE $DESCRIPTION: "invert source mask"
} dt_iop_ekvidokazuki_params_t;

/* Module-level: which module's raster-mask users table kazuki registered in (spec §5.7). */
typedef struct dt_iop_ekvidokazuki_instance_t
{
  dt_iop_module_t *registered_source;
} dt_iop_ekvidokazuki_instance_t;

typedef struct dt_iop_ekvidokazuki_gui_data_t
{
  dt_gui_collapsible_section_t detect_section, colour_section, halo_section, haze_section, star_section;
  GtkDrawingArea *area;          // kernel preview pane
  unsigned char *view;           // cached BGRA kernel image, view_width^2
  int view_width;
  gboolean view_dirty;
  float view_diag;               // diagonal (px) the view was built at, for the scale bar
  GtkWidget *preview;            // GUI-only combobox: final / glow only / energy / warmth / clipped
  ekvido_kazuki_view_t show;     // GUI-only: what the darkroom shows
  GtkWidget *blend_white_ev, *clip_ev;
  float clip_level;              // auto clip level measured by the preview pipe (NAN = none yet)
  dt_hash_t clip_hash;           // pipe hash the preview pipe measured it at
  GtkWidget *source_combo;       // source mask picker (like blending's raster-mask picker)
  gboolean source_missing;       // full pipe: a source is chosen but its mask is unavailable
  guint trouble_idle;            // pending main-thread update of the module warning
} dt_iop_ekvidokazuki_gui_data_t;

typedef struct dt_iop_ekvidokazuki_data_t
{
  dt_iop_ekvidokazuki_params_t p;
  ekvido_kazuki_cache_t *cache;
#ifdef HAVE_OPENCL
  ekv_cl_cache_t *clcache;       // the GPU's cached glow (spec D7)
#endif
  dt_iop_module_t *source;       // resolved source mask module (NULL = none or not found)
} dt_iop_ekvidokazuki_data_t;

int legacy_params(dt_iop_module_t *self, const void *const old_params, const int old_version, void **new_params,
                  int32_t *new_params_size, int *new_version)
{
  if(old_version == 1)
  {
    /* v1 is exactly v2's first fields; v2 adds the source mask, "none" by default */
    dt_iop_ekvidokazuki_params_t *n = calloc(1, sizeof(dt_iop_ekvidokazuki_params_t));
    if(!n) return 1;
    memcpy(n, old_params, offsetof(dt_iop_ekvidokazuki_params_t, source_mask_op));
    n->source_mask_id = BLEND_RASTER_ID;
    *new_params = n;
    *new_params_size = sizeof(dt_iop_ekvidokazuki_params_t);
    *new_version = 2;
    return 0;
  }
  return 1;
}

void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);
  self->data = calloc(1, sizeof(dt_iop_ekvidokazuki_instance_t));
}

static gboolean _in_dev(const dt_iop_module_t *self, const dt_iop_module_t *m)
{
  if(!m || !self->dev) return FALSE;
  for(const GList *l = self->dev->iop; l; l = g_list_next(l))
    if(l->data == m) return TRUE;
  return FALSE;
}

/* Register kazuki as a user of `src`'s raster mask so darktable stores it (as blending does),
   and drop the previous registration unless kazuki's own blending still uses that module.
   The users table maps user module -> mask id; darktable treats its keys as modules.
   Known limitations (accepted):
   - kazuki's blending registers under the same key (kazuki). If both use the SAME source module
     with DIFFERENT mask ids, whichever commits last wins and the other mask is not stored (rare:
     ids are almost always BLEND_RASTER_ID). The order is blend commit, then kazuki's commit_params.
   - source_mask_instance does not follow renumbering of the source's instances (reorder, deleting
     instance 0, undo/redo): the source then shows "(unavailable)", the warning appears and the
     user re-picks it. dt_iop_update_multi_priority() (imageop.c, ekvido hunk) leaves kazuki's
     blend params alone unless its blending references that source. */
static void _register_source(dt_iop_module_t *self, dt_iop_module_t *src, const dt_mask_id_t id)
{
  dt_iop_ekvidokazuki_instance_t *inst = self->data;
  if(!inst) return;
  dt_iop_module_t *old = inst->registered_source;
  if(old && old != src && old != self->raster_mask.sink.source && _in_dev(self, old))
    g_hash_table_remove(old->raster_mask.source.users, self);
  if(src) g_hash_table_insert(src->raster_mask.source.users, self, GINT_TO_POINTER(id));
  inst->registered_source = src;
}

void cleanup(dt_iop_module_t *self)
{
  _register_source(self, NULL, BLEND_RASTER_ID);
  free(self->data);
  self->data = NULL;
  dt_iop_default_cleanup(self);
}

/* The module named by the params (op + instance), if it is in this edit and before kazuki. */
static dt_iop_module_t *_find_source(dt_iop_module_t *self, const dt_iop_ekvidokazuki_params_t *p)
{
  if(!p->source_mask_op[0] || !self->dev) return NULL;
  for(GList *l = self->dev->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(m == self) break;
    if(dt_iop_module_is(m, p->source_mask_op) && m->multi_priority == p->source_mask_instance) return m;
  }
  return NULL;
}

/* The source module's raster mask, carried through every distorting module between it and
   kazuki to kazuki's own input region (roi_in, margin included). NULL when unavailable.
   Same rules as dt_dev_get_raster_mask(), which only delivers masks at a module's roi_out. */
/* MUST mirror _skip_piece_on_tags() in src/develop/pixelpipe_hb.c (static there): a piece the pipe
   skipped has stale processed ROIs and raster masks. Rebase checklist: re-check it on every upstream merge. */
static gboolean _piece_skipped(const dt_dev_pixelpipe_iop_t *piece)
{
  if(!piece->enabled || piece->module->iop_order == INT_MAX) return TRUE;
  return dt_iop_module_is_skipped(piece->module->dev, piece->module) && dt_pipe_is_basic(piece->pipe);
}

static float *_source_mask(const dt_iop_ekvidokazuki_data_t *d, dt_dev_pixelpipe_iop_t *piece,
                           const dt_iop_roi_t *roi_in, gboolean *free_mask)
{
  *free_mask = FALSE;
  if(!d->source) return NULL;
  GList *src_node = NULL;
  for(GList *l = piece->pipe->nodes; l && l->data != piece; l = g_list_next(l))
    if(((dt_dev_pixelpipe_iop_t *)l->data)->module == d->source) src_node = l;
  if(!src_node) return NULL;
  dt_dev_pixelpipe_iop_t *src = src_node->data;
  const gboolean writes = src->module->blend_params->mask_mode > DEVELOP_MASK_ENABLED
                          || (src->module->flags() & IOP_FLAGS_WRITE_RASTER);
  if(_piece_skipped(src) || !writes) return NULL;
  float *cur = g_hash_table_lookup(src->raster_masks, GINT_TO_POINTER(d->p.source_mask_id));
  if(!cur) return NULL;
  dt_iop_roi_t roi = src->processed_roi_out;
  for(GList *l = g_list_next(src_node); l && l->data != piece; l = g_list_next(l))
  {
    dt_dev_pixelpipe_iop_t *it = l->data;
    if(_piece_skipped(it)) continue;
    /* the buffer in hand must be the size this piece read; anything else is a stale ROI */
    if(it->processed_roi_in.width != roi.width || it->processed_roi_in.height != roi.height)
    {
      if(*free_mask) dt_free_align(cur);
      *free_mask = FALSE;
      return NULL;
    }
    if(it->module->distort_mask)
    {
      float *next = dt_iop_image_alloc(it->processed_roi_out.width, it->processed_roi_out.height, 1);
      if(next)
        it->module->distort_mask(it->module, it, cur, next, &it->processed_roi_in, &it->processed_roi_out);
      if(*free_mask) dt_free_align(cur);
      if(!next) { *free_mask = FALSE; return NULL; }
      cur = next;
      *free_mask = TRUE;
      roi = it->processed_roi_out;
    }
    else if(memcmp(&it->processed_roi_in, &it->processed_roi_out, sizeof(dt_iop_roi_t)))
    {
      if(*free_mask) dt_free_align(cur); /* changes the geometry but cannot carry masks */
      *free_mask = FALSE;
      return NULL;
    }
  }
  if(roi.width != roi_in->width || roi.height != roi_in->height)
  {
    if(*free_mask) dt_free_align(cur);
    *free_mask = FALSE;
    return NULL;
  }
  return cur;
}

static gboolean _source_trouble_idle(gpointer user_data)
{
  dt_iop_module_t *self = user_data;
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  if(!g) return G_SOURCE_REMOVE;
  dt_iop_gui_enter_critical_section(self);
  const gboolean missing = g->source_missing;
  g->trouble_idle = 0;
  dt_iop_gui_leave_critical_section(self);
  if(missing)
  {
    const dt_iop_ekvidokazuki_params_t *p = self->params;
    gchar *msg = g_strdup_printf(_("source mask from %s unavailable: no glow"),
                                 dt_iop_get_localized_name(p->source_mask_op));
    dt_iop_set_module_trouble_message(self, msg,
                                      _("the source mask module is missing, disabled, after kazuki in the pipe, "
                                        "or has no drawn or parametric mask.\nkazuki adds no glow until that is "
                                        "fixed or the source mask is set to none."),
                                      NULL);
    g_free(msg);
  }
  else
    dt_iop_set_module_trouble_message(self, NULL, NULL, NULL);
  return G_SOURCE_REMOVE;
}

/* Record whether the chosen source mask was available; the warning itself is shown from the
   main thread. Without a GUI (export, darktable-cli) a missing mask is printed to the log. */
static void _note_source(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const gboolean missing)
{
  const dt_iop_ekvidokazuki_data_t *d = piece->data;
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  if(!(g && self->dev->gui_attached))
  {
    if(missing)
      dt_print(DT_DEBUG_ALWAYS, "[ekvidokazuki] source mask from '%s' unavailable: no glow", d->p.source_mask_op);
    return;
  }
  if(!dt_pipe_is_full(piece->pipe)) return;
  dt_iop_gui_enter_critical_section(self);
  const gboolean changed = g->source_missing != missing;
  g->source_missing = missing;
  if(changed && !g->trouble_idle) g->trouble_idle = g_idle_add(_source_trouble_idle, self);
  dt_iop_gui_leave_critical_section(self);
}

/* kazuki's output is a crop of its input (zoom margin), so masks are cropped the same way;
   without this, raster masks used further down the pipe fail whenever kazuki adds a margin. */
void distort_mask(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const float *const in, float *const out,
                  const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  dt_iop_copy_image_roi(out, in, 1, roi_in, roi_out);
}

const char *name()
{
  return _("ekvido kazuki");
}

const char *aliases()
{
  return _("kazuki|glow|bloom|halation|star|glint|flare");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("highlights glow in their own warm or cool colour, with halo, haze and star shapes"),
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

/* Fills a caller-owned core struct (never return the address of a compound literal). */
static void _core_params(const dt_iop_ekvidokazuki_params_t *p, ekvido_kazuki_params_t *k)
{
  memset(k, 0, sizeof(*k));
  k->strength = p->strength;
  k->blend = (ekvido_blend_mode_t)p->blend;
  k->blend_white_ev = p->blend_white_ev;
  k->threshold_ev = p->threshold_ev;
  k->knee_ev = p->knee_ev;
  k->hue_radius = p->hue_radius;
  k->neutral_point = p->neutral_point;
  k->warmth_softness = p->warmth_softness;
  k->neutral_bias = p->neutral_bias;
  k->push_warm = p->push_warm;
  k->push_cool = p->push_cool;
  ekvido_hsv_to_rgb(p->warm_core_hue, p->warm_core_sat, k->warm_core);
  ekvido_hsv_to_rgb(p->warm_edge_hue, p->warm_edge_sat, k->warm_edge);
  ekvido_hsv_to_rgb(p->cool_core_hue, p->cool_core_sat, k->cool_core);
  ekvido_hsv_to_rgb(p->cool_edge_hue, p->cool_edge_sat, k->cool_edge);
  k->band_split = p->band_split;
  k->spec = (ekvido_kernel_spec_t){ .halo_radius = p->halo_radius, .halo_beta = p->halo_beta,
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

/* What this pipe shows: previews only in the darkroom's main view, never in exports. */
static ekvido_kazuki_view_t _view(const dt_iop_module_t *self, const dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  return g && self->dev->gui_attached && dt_pipe_is_full(piece->pipe) ? g->show : EKVIDO_VIEW_FINAL;
}

void modify_roi_in(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *const roi_out,
                   dt_iop_roi_t *roi_in)
{
  *roi_in = *roi_out;
  const dt_iop_ekvidokazuki_data_t *d = piece->data;
  ekvido_kazuki_params_t k;
  _core_params(&d->p, &k);
  const gboolean need = d->p.strength > 0.f || _view(self, piece) != EKVIDO_VIEW_FINAL;
  const int margin = need ? ekvido_kazuki_margin(&k, _diag_px(piece, roi_out->scale)) : 0;
  const ekvido_rect_t r = ekvido_expand_rect(
      (ekvido_rect_t){ roi_out->x, roi_out->y, roi_out->width, roi_out->height }, margin,
      (int)floorf(piece->buf_in.width * roi_out->scale), (int)floorf(piece->buf_in.height * roi_out->scale));
  roi_in->x = r.x;
  roi_in->y = r.y;
  roi_in->width = r.w;
  roi_in->height = r.h;
}

/* Where the whole-frame clip estimate reads from: the host input, or (GPU) the device image. */
typedef struct _clip_src_t
{
  const float *host;
  void *dev; /* cl_mem */
  void *gpu; /* ekv_cl_t * */
  int w, h;
} _clip_src_t;

static float _clip_estimate(const _clip_src_t *s)
{
#ifdef HAVE_OPENCL
  if(s->gpu) return ekv_cl_estimate_clip_level((ekv_cl_t *)s->gpu, (cl_mem)s->dev, s->w, s->h);
#endif
  return ekvido_estimate_clip_level(s->host, (size_t)s->w * s->h, 4);
}

/* Clip level (spec §6.4): manual = 2^clip_ev; auto = Python's plateau estimate whenever the
   module sees the whole frame (preview pipe, fit view, export); a zoomed full pipe takes the
   preview pipe's value (hazeremoval pattern). The estimate runs on the CPU or the GPU (_clip_src_t). A failed GPU estimate
   (NAN) is not published, so a zoomed full pipe keeps waiting for the CPU fallback's value. */
static float _clip_level(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const _clip_src_t *src,
                         const dt_iop_roi_t *roi_in)
{
  const dt_iop_ekvidokazuki_data_t *d = piece->data;
  if(d->p.clip_mode == EKVIDOKAZUKI_CLIP_MANUAL) return exp2f(d->p.clip_ev);
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  const gboolean gui = g && self->dev->gui_attached;
  const int full_w = (int)floorf(piece->buf_in.width * roi_in->scale);
  const int full_h = (int)floorf(piece->buf_in.height * roi_in->scale);
  const gboolean whole = roi_in->x == 0 && roi_in->y == 0 && roi_in->width >= full_w && roi_in->height >= full_h;
  float clip = NAN;
  if(gui && !whole && dt_pipe_is_full(piece->pipe))
  {
    dt_iop_gui_enter_critical_section(self);
    const dt_hash_t hash = g->clip_hash;
    dt_iop_gui_leave_critical_section(self);
    if(hash != DT_INVALID_HASH
       && !dt_dev_sync_pixelpipe_hash(self->dev, piece->pipe, self->iop_order, DT_DEV_TRANSFORM_DIR_BACK_INCL,
                                      &self->gui_lock, &g->clip_hash))
      dt_control_log(_("inconsistent output"));
    dt_iop_gui_enter_critical_section(self);
    clip = g->clip_level;
    dt_iop_gui_leave_critical_section(self);
  }
  if(dt_isnan(clip)) clip = _clip_estimate(src);
  if(gui && dt_pipe_is_preview(piece->pipe) && !dt_isnan(clip))
  {
    const dt_hash_t hash = dt_dev_hash_plus(self->dev, piece->pipe, self->iop_order, DT_DEV_TRANSFORM_DIR_BACK_INCL);
    dt_iop_gui_enter_critical_section(self);
    g->clip_level = clip;
    g->clip_hash = hash;
    dt_iop_gui_leave_critical_section(self);
  }
  return clip;
}

void process(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid, void *const ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4, self, piece->colors, ivoid, ovoid, roi_in, roi_out)) return;
  dt_iop_ekvidokazuki_data_t *d = piece->data;
  ekvido_kazuki_params_t k;
  _core_params(&d->p, &k);
  const ekvido_kazuki_view_t view = _view(self, piece);
  const gboolean interactive = self->dev->gui_attached && !(piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT);
  const _clip_src_t cs = { .host = ivoid, .w = roi_in->width, .h = roi_in->height };
  const float clip = _clip_level(self, piece, &cs, roi_in);
  const ekvido_rect_t rect = { roi_out->x - roi_in->x, roi_out->y - roi_in->y, roi_out->width, roi_out->height };
  gboolean free_mask = FALSE;
  float *mask = NULL;
  if(d->p.source_mask_op[0])
  {
    mask = _source_mask(d, piece, roi_in, &free_mask);
    _note_source(self, piece, mask == NULL);
    if(!mask) /* spec §5.7: a chosen but unavailable source mask means no glow */
    {
      dt_iop_copy_image_roi(ovoid, ivoid, 4, roi_in, roi_out);
      return;
    }
    dt_dump_pfm("source_mask", mask, roi_in->width, roi_in->height, sizeof(float), self->op);
  }
  else
    _note_source(self, piece, FALSE);
  const int err = ekvido_kazuki_process_masked(ivoid, roi_in->width, roi_in->height, ovoid, rect, &k,
                                               _diag_px(piece, roi_in->scale), clip, mask, d->p.source_mask_invert,
                                               interactive, view, interactive ? d->cache : NULL,
                                               (int)dt_get_num_threads());
  if(free_mask) dt_free_align(mask);
  if(err)
  {
    dt_iop_copy_image_roi(ovoid, ivoid, 4, roi_in, roi_out);
    dt_control_log(_("ekvido kazuki: not enough memory, module skipped"));
  }
}

#ifdef HAVE_OPENCL
/* GPU twin of process() (spec 2026-10-16 OpenCL §4.4); any failure returns an error and darktable
   runs process() for this module instead (D8). dev_in is never written. */
int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4, self, piece->colors, (void *)dev_in, (void *)dev_out, roi_in, roi_out))
    return DT_OPENCL_PROCESS_CL;
  ekv_cl_t *E = ekv_cl_dt_get(piece->pipe->devid);
  if(!E) return DT_OPENCL_PROCESS_CL;
  dt_iop_ekvidokazuki_data_t *d = piece->data;
  ekvido_kazuki_params_t k;
  _core_params(&d->p, &k);
  const ekvido_kazuki_view_t view = _view(self, piece);
  const gboolean interactive = self->dev->gui_attached && !(piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT);
  const _clip_src_t cs = { .dev = dev_in, .gpu = E, .w = roi_in->width, .h = roi_in->height };
  const float clip = _clip_level(self, piece, &cs, roi_in);
  if(dt_isnan(clip)) return DT_OPENCL_DEFAULT_ERROR;
  const ekvido_rect_t rect = { roi_out->x - roi_in->x, roi_out->y - roi_in->y, roi_out->width, roi_out->height };
  gboolean free_mask = FALSE;
  float *mask = NULL;
  if(d->p.source_mask_op[0])
  {
    mask = _source_mask(d, piece, roi_in, &free_mask);
    _note_source(self, piece, mask == NULL);
    if(!mask) /* spec §5.7: a chosen but unavailable source mask means no glow */
      return ekv_cl_copy_rect(E, dev_in, dev_out, rect) ? DT_OPENCL_DEFAULT_ERROR : CL_SUCCESS;
    dt_dump_pfm("source_mask", mask, roi_in->width, roi_in->height, sizeof(float), self->op);
  }
  else
    _note_source(self, piece, FALSE);
  const int err = ekv_cl_kazuki_process(E, dev_in, roi_in->width, roi_in->height, dev_out, rect, &k,
                                        _diag_px(piece, roi_in->scale), clip, mask, d->p.source_mask_invert,
                                        interactive, view, interactive ? d->clcache : NULL);
  if(free_mask) dt_free_align(mask);
  return err ? DT_OPENCL_DEFAULT_ERROR : CL_SUCCESS;
}

/* Honest GPU need (spec §4.4): everything through factor_cl — tiling->overhead is a 32-bit
   unsigned, too small at 100 MP. Not tiled (no IOP_FLAGS_ALLOW_TILING): if it does not fit,
   darktable runs process() on the CPU without an error. */
void tiling_callback(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out, dt_develop_tiling_t *tiling)
{
  const dt_iop_ekvidokazuki_data_t *d = piece->data;
  ekvido_kazuki_params_t k;
  _core_params(&d->p, &k);
  const float diag = _diag_px(piece, roi_in->scale);
  const gboolean interactive = self->dev->gui_attached && !(piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT);
  const int pad = interactive ? ekvido_kernel_half_cap(diag) : ekvido_kernel_half_size(&k.spec, diag);
  const double px = (double)roi_in->width * roi_in->height;
  const double extra = (double)ekv_cl_kazuki_bytes(roi_in->width, roi_in->height, pad)
                       + (interactive ? 3.0 * sizeof(float) * px : 0.0); /* the cached glow */
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
  dt_iop_ekvidokazuki_data_t *d = calloc(1, sizeof(dt_iop_ekvidokazuki_data_t));
  d->cache = ekvido_kazuki_cache_new();
#ifdef HAVE_OPENCL
  d->clcache = ekv_cl_cache_new();
#endif
  piece->data = d;
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_ekvidokazuki_data_t *d = piece->data;
  if(d)
  {
    ekvido_kazuki_cache_free(d->cache);
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
  dt_iop_ekvidokazuki_data_t *d = piece->data;
  memcpy(&d->p, p1, sizeof(d->p));
  dt_iop_module_t *src = _find_source(self, &d->p);
  _register_source(self, src, d->p.source_mask_id);
  /* the source's cached output may predate kazuki asking for its mask: recompute from there */
  if(src && src != d->source) dt_dev_pixelpipe_cache_invalidate_later(pipe, src->iop_order, "ekvidokazuki source mask: ");
  d->source = src;
}

static dt_iop_ekvidokazuki_params_t _defaults(void)
{
  return (dt_iop_ekvidokazuki_params_t){
    .strength = 0.5f, .blend = EKVIDOKAZUKI_BLEND_ADDITIVE, .blend_white_ev = 0.f, .threshold_ev = 0.f,
    .knee_ev = 1.f, .clip_mode = EKVIDOKAZUKI_CLIP_AUTO, .clip_ev = 0.f, .hue_radius = 0.01f,
    .neutral_point = 0.f, .warmth_softness = 0.05f, .neutral_bias = 0.8f, .push_warm = 0.7f, .push_cool = 0.7f,
    .warm_core_hue = 40.f, .warm_core_sat = 0.45f, .warm_edge_hue = 22.5f, .warm_edge_sat = 0.88f,
    .cool_core_hue = 193.846154f, .cool_core_sat = 0.65f, .cool_edge_hue = 232.173913f, .cool_edge_sat = 0.92f,
    .band_split = 0.006f, .halo_radius = 0.006f, .halo_beta = 1.3f, .halo_stretch_x = 1.85f,
    .halo_stretch_y = 1.f, .veil_radius = 0.1f, .veil_beta = 1.2f, .w_halo = 1.f, .w_veil = 0.f, .arms = 4,
    .arm_angle = 0.f, .arm_radius = 0.001f, .arm_beta = 0.96f, .arm_sharpness = 0.32f, .arm_stretch_x = 2.93f,
    .arm_stretch_y = 0.58f, .w_arms = 0.2f };
}

/* Shapes only (by design): thresholds, neutral point, strength and colours stay at defaults. */
void init_presets(dt_iop_module_so_t *self)
{
  dt_iop_ekvidokazuki_params_t p = _defaults();
  dt_gui_presets_add_generic(_("kazuki default"), self->op, self->version(), &p, sizeof(p), TRUE,
                             DEVELOP_BLEND_CS_RGB_SCENE);
  p.w_halo = 0.f;
  p.w_veil = 0.f;
  p.band_split = 0.f;
  p.arms = 4;
  p.arm_angle = 0.f;
  p.arm_radius = 0.02f;
  p.arm_beta = 4.f;
  p.arm_sharpness = 1.23f;
  p.arm_stretch_x = 4.f;
  p.arm_stretch_y = 1.71f;
  p.w_arms = 1.f;
  dt_gui_presets_add_generic(_("kazuki wide"), self->op, self->version(), &p, sizeof(p), TRUE,
                             DEVELOP_BLEND_CS_RGB_SCENE);
}

static GtkWidget *_slider(dt_iop_module_t *sect, const char *param, const char *tooltip, int digits)
{
  GtkWidget *w = dt_bauhaus_slider_from_params(sect, param);
  if(digits >= 0) dt_bauhaus_slider_set_digits(w, digits);
  gtk_widget_set_tooltip_text(w, tooltip);
  return w;
}

/* Hue slider with a rainbow track (display colours only). */
static GtkWidget *_hue_slider(dt_iop_module_t *sect, const char *param, const char *tooltip)
{
  GtkWidget *w = _slider(sect, param, tooltip, 1);
  dt_bauhaus_slider_set_format(w, "°");
  for(int i = 0; i <= 6; i++)
  {
    float rgb[3];
    ekvido_hsv_to_rgb(60.f * i, 1.f, rgb);
    dt_bauhaus_slider_set_stop(w, i / 6.f, rgb[0], rgb[1], rgb[2]);
  }
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
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  GtkAllocation a;
  gtk_widget_get_allocation(widget, &a);
  const int width = a.width;
  if(width < 16) return FALSE;
  if(width != g->view_width || g->view_dirty || !g->view)
  {
    free(g->view);
    g->view = malloc((size_t)width * width * 4);
    if(!g->view) { g->view_width = 0; return FALSE; }
    ekvido_kazuki_params_t k;
    _core_params(self->params, &k);
    g->view_diag = ekvido_kernel_view(&k.spec, width, g->view);
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

static void _set_show(dt_iop_module_t *self, ekvido_kazuki_view_t show)
{
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  if(g->show == show) return;
  g->show = show;
  ++darktable.gui->reset;
  dt_bauhaus_combobox_set(g->preview, (int)show);
  --darktable.gui->reset;
  dt_iop_refresh_center(self);
}

/* GUI-only preview: never written to history (the combobox is not a param). */
static void _preview_changed(GtkWidget *combo, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  const ekvido_kazuki_view_t show = (ekvido_kazuki_view_t)dt_bauhaus_combobox_get(combo);
  if(show != EKVIDO_VIEW_FINAL && self->request_mask_display)
  {
    dt_control_log(_("cannot show a glow preview while a blend mask is displayed"));
    ++darktable.gui->reset;
    dt_bauhaus_combobox_set(combo, (int)EKVIDO_VIEW_FINAL);
    --darktable.gui->reset;
    if(g->show != EKVIDO_VIEW_FINAL)
    {
      g->show = EKVIDO_VIEW_FINAL;
      dt_iop_refresh_center(self);
    }
    return;
  }
  if(show != EKVIDO_VIEW_FINAL)
  {
    dt_iop_request_focus(self);
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE); /* enable the module */
    for(GList *l = self->dev->iop; l; l = g_list_next(l))   /* only one instance previews at a time */
    {
      dt_iop_module_t *m = l->data;
      if(m != self && m->gui_data && !strcmp(m->op, self->op)) _set_show(m, EKVIDO_VIEW_FINAL);
    }
  }
  g->show = show;
  dt_iop_refresh_center(self);
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  if(!in && self->gui_data) _set_show(self, EKVIDO_VIEW_FINAL);
}

static void _update_sensitivity(dt_iop_module_t *self)
{
  const dt_iop_ekvidokazuki_params_t *p = self->params;
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  gtk_widget_set_sensitive(g->blend_white_ev, p->blend >= EKVIDOKAZUKI_BLEND_SCREEN);
  gtk_widget_set_sensitive(g->clip_ev, p->clip_mode == EKVIDOKAZUKI_CLIP_MANUAL);
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  _update_sensitivity(self);
  g->view_dirty = TRUE;
  gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

typedef struct _source_entry_t
{
  dt_iop_module_t *module;
  dt_mask_id_t id;
} _source_entry_t;

/* Mirrors blending's raster-mask picker: "none" plus every mask offered by a module before kazuki. */
static void _source_combo_populate(GtkWidget *w, dt_iop_module_t **m)
{
  dt_iop_module_t *self = *m;
  const dt_iop_ekvidokazuki_params_t *p = self->params;
  ++darktable.gui->reset;
  dt_bauhaus_combobox_clear(w);
  _source_entry_t *entry = calloc(1, sizeof(_source_entry_t));
  dt_bauhaus_combobox_add_full(w, _("none"), DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT, entry, free, TRUE);
  int pos = 0, i = 1;
  for(GList *l = self->dev->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *iop = l->data;
    if(iop == self) break;
    GHashTableIter it;
    gpointer key, value;
    g_hash_table_iter_init(&it, iop->raster_mask.source.masks);
    while(g_hash_table_iter_next(&it, &key, &value))
    {
      entry = calloc(1, sizeof(_source_entry_t));
      entry->module = iop;
      entry->id = GPOINTER_TO_INT(key);
      dt_bauhaus_combobox_add_full(w, (const char *)value, DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT, entry, free, TRUE);
      if(dt_iop_module_is(iop, p->source_mask_op) && iop->multi_priority == p->source_mask_instance
         && entry->id == p->source_mask_id)
        pos = i;
      i++;
    }
  }
  if(p->source_mask_op[0] && pos == 0) /* chosen source not offered (removed, disabled, no mask) */
  {
    gchar *label = g_strdup_printf(_("%s (unavailable)"), dt_iop_get_localized_name(p->source_mask_op));
    entry = calloc(1, sizeof(_source_entry_t));
    entry->id = INVALID_MASKID; /* selecting it keeps the params as they are */
    dt_bauhaus_combobox_add_full(w, label, DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT, entry, free, TRUE);
    g_free(label);
    pos = i;
  }
  dt_bauhaus_combobox_set(w, pos);
  --darktable.gui->reset;
}

static void _source_changed(GtkWidget *w, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  const _source_entry_t *entry = dt_bauhaus_combobox_get_data(w);
  if(!entry || (!entry->module && entry->id == INVALID_MASKID)) return;
  dt_iop_ekvidokazuki_params_t *p = self->params;
  gboolean reprocess = FALSE;
  memset(p->source_mask_op, 0, sizeof(p->source_mask_op));
  p->source_mask_instance = 0;
  p->source_mask_id = BLEND_RASTER_ID;
  if(entry->module)
  {
    g_strlcpy(p->source_mask_op, entry->module->op, sizeof(p->source_mask_op));
    p->source_mask_instance = entry->module->multi_priority;
    p->source_mask_id = entry->id;
    reprocess = !dt_iop_is_raster_mask_used(entry->module, entry->id);
    _register_source(self, entry->module, entry->id);
  }
  dt_dev_add_history_item(self->dev, self, TRUE);
  if(reprocess) dt_dev_reprocess_all(self->dev);
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  if(g->trouble_idle) g_source_remove(g->trouble_idle);
  g->trouble_idle = 0;
  free(g->view);
  g->view = NULL;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_ekvidokazuki_gui_data_t *g = IOP_GUI_ALLOC(ekvidokazuki);
  g->clip_level = NAN;
  g->clip_hash = DT_INVALID_HASH;
  g->show = EKVIDO_VIEW_FINAL;
  g->area = GTK_DRAWING_AREA(dtgtk_drawing_area_new_with_height(0));
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->area),
                              _("glow shape on a log scale; the bar shows a fraction of the image diagonal"));
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_draw_kernel), self);
  DT_BAUHAUS_COMBOBOX_NEW_FULL(g->preview, self, NULL, N_("preview"),
                               _("what the centre view shows (not saved, not exported): the final image, "
                                 "the glow alone, the highlight energy, the warmth map (blue cool, grey "
                                 "neutral, orange warm) or the clipped areas in red"),
                               0, _preview_changed, self, N_("final"), N_("glow only"), N_("energy"),
                               N_("warmth"), N_("clipped"));
  g->view_dirty = TRUE;
  self->widget = dt_gui_vbox(GTK_WIDGET(g->area), g->preview);

  _slider(self, "strength", _("overall amount of glow added"), 3);
  GtkWidget *blend = dt_bauhaus_combobox_from_params(self, "blend");
  gtk_widget_set_tooltip_text(blend, _("additive: glow adds light, like flare. conserve: light moves from sources "
                                       "to surroundings, total stays the same; screen / overlay / colour dodge / "
                                       "glow dodge: Clip Studio Paint's layer modes (overlay also darkens "
                                       "glow-free areas); overlay (glow only): overlay that leaves glow-free "
                                       "areas alone"));
  g->blend_white_ev = _slider(self, "blend_white_ev",
                              _("for screen, the overlays and the dodges: the brightness treated as the layer's white; "
                                "0 = paper white"), 2);

  dt_iop_module_t sec;
  dt_iop_module_t *s = _section(self, &sec, &g->detect_section, "plugins/darkroom/ekvidokazuki/expand_detection",
                                N_("detection"));
  g->source_combo = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->source_combo, N_("detection"), N_("source mask"));
  dt_bauhaus_combobox_add_populate_fct(g->source_combo, _source_combo_populate);
  g_signal_connect(G_OBJECT(g->source_combo), "value-changed", G_CALLBACK(_source_changed), self);
  gtk_widget_set_tooltip_text(g->source_combo,
                              _("where light may come from: the drawn and/or parametric mask of a module "
                                "before kazuki (e.g. an exposure instance at 0 EV). only masked areas emit; "
                                "the glow still spreads over the whole image. kazuki's own blending mask "
                                "still gates the output."));
  gtk_box_pack_start(GTK_BOX(g->detect_section.container), g->source_combo, TRUE, TRUE, 0);
  GtkWidget *invert = dt_bauhaus_toggle_from_params(s, "source_mask_invert");
  gtk_widget_set_tooltip_text(invert, _("light comes from outside the source mask instead"));
  _slider(s, "threshold_ev", _("how bright a pixel must be to glow; 0 = paper white, each +1 doubles it"), 2);
  _slider(s, "knee_ev", _("fades glow in gradually around the threshold instead of a hard cut"), 2);
  GtkWidget *clip_mode = dt_bauhaus_combobox_from_params(s, "clip_mode");
  gtk_widget_set_tooltip_text(clip_mode, _("auto: find clipped highlights automatically; manual: use the clip level below"));
  g->clip_ev = _slider(s, "clip_ev", _("in manual mode, the brightness treated as clipped; 0 = paper white"), 2);

  s = _section(self, &sec, &g->colour_section, "plugins/darkroom/ekvidokazuki/expand_colour", N_("colour"));
  _slider(s, "hue_radius", _("how far around each bright spot its colour is read from; clipped spots take their fringe's colour"), 3);
  _slider(s, "neutral_point", _("shifts what counts as neutral; move it if neutral sources glow orange or blue"), 3);
  _slider(s, "warmth_softness", _("how gradually glows change from cool to warm; higher gives more in-between tones"), 3);
  _slider(s, "neutral_bias", _("warmth for blown-out spots with no colour around them: 1 warm, 0 cool"), 2);
  _slider(s, "push_warm", _("how strongly warm glows take the warm colours; 0 keeps the source's own colour"), 2);
  _slider(s, "push_cool", _("how strongly cool glows take the cool colours; 0 keeps the source's own colour"), 2);
  _hue_slider(s, "warm_core_hue", _("colour at the centre of warm glows (pale gold)"));
  _slider(s, "warm_core_sat", _("colour at the centre of warm glows (pale gold)"), 2);
  _hue_slider(s, "warm_edge_hue", _("colour at the outside of warm glows (orange)"));
  _slider(s, "warm_edge_sat", _("colour at the outside of warm glows (orange)"), 2);
  _hue_slider(s, "cool_core_hue", _("colour at the centre of cool glows (cyan)"));
  _slider(s, "cool_core_sat", _("colour at the centre of cool glows (cyan)"), 2);
  _hue_slider(s, "cool_edge_hue", _("colour at the outside of cool glows (deep blue)"));
  _slider(s, "cool_edge_sat", _("colour at the outside of cool glows (deep blue)"), 2);
  _slider(s, "band_split", _("where the centre colour hands over to the edge colour"), 4);

  s = _section(self, &sec, &g->halo_section, "plugins/darkroom/ekvidokazuki/expand_halo", N_("halo"));
  _slider(s, "halo_radius", _("size of the main soft glow (fraction of the image diagonal)"), 4);
  _slider(s, "halo_beta", _("low: long misty tail; high: tight glow"), 2);
  _slider(s, "halo_stretch_x", _("stretch the halo into an oval"), 2);
  _slider(s, "halo_stretch_y", _("stretch the halo into an oval"), 2);
  _slider(s, "w_halo", _("share of the glow in the halo (black mist: high)"), 2);

  s = _section(self, &sec, &g->haze_section, "plugins/darkroom/ekvidokazuki/expand_haze", N_("haze"));
  _slider(s, "veil_radius", _("size of the very wide, faint haze"), 3);
  _slider(s, "veil_beta", _("low: haze reaches further"), 2);
  _slider(s, "w_veil", _("share of the glow in the wide haze (white mist: high)"), 2);

  s = _section(self, &sec, &g->star_section, "plugins/darkroom/ekvidokazuki/expand_star", N_("star"));
  _slider(s, "arms", _("number of arms; 4 = cross, 2 = horizontal streak, 0 = off"), -1);
  GtkWidget *angle = _slider(s, "arm_angle", _("angle of the first arm (degrees)"), 1);
  dt_bauhaus_slider_set_format(angle, "°");
  _slider(s, "arm_radius", _("size of the star's bright centre (fraction of the image diagonal)"), 4);
  _slider(s, "arm_beta", _("low: long arms on bright sources; high: short arms"), 2);
  _slider(s, "arm_sharpness", _("2 round, 1 diamond, below 1 thinner arms"), 2);
  _slider(s, "arm_stretch_x", _("stretch the whole star"), 2);
  _slider(s, "arm_stretch_y", _("stretch the whole star"), 2);
  _slider(s, "w_arms", _("share of the glow in the star (low: only the hottest points show arms)"), 2);
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_ekvidokazuki_gui_data_t *g = self->gui_data;
  dt_gui_update_collapsible_section(&g->detect_section);
  dt_gui_update_collapsible_section(&g->colour_section);
  dt_gui_update_collapsible_section(&g->halo_section);
  dt_gui_update_collapsible_section(&g->haze_section);
  dt_gui_update_collapsible_section(&g->star_section);
  dt_iop_gui_enter_critical_section(self);
  g->clip_level = NAN;
  g->clip_hash = DT_INVALID_HASH;
  dt_iop_gui_leave_critical_section(self);
  _source_combo_populate(g->source_combo, &self);
  _update_sensitivity(self);
  g->view_dirty = TRUE;
  gtk_widget_queue_draw(GTK_WIDGET(g->area));
}
