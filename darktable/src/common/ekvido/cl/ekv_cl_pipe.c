#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_OPENCL
/* The Phase 1 streaming pipelines on the GPU (kazuki.c _glow_stream, mist.c _source/_glow) and the
   per-piece glow cache (spec D7): only the final 3 glow planes are kept, keyed on everything the
   glow depends on, so strength / blend mode / blend white skip every FFT. */
#include "common/ekvido/cl/ekv_cl_priv.h"
#include "common/ekvido/cl/ekv_clredirect.h"
#include "common/ekvido/convolve.h"
#include "common/ekvido/hash.h"
#include "common/ekvido/psf.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EKV_KEY_MAX 512

struct ekv_cl_cache_t
{
  ekv_cl_t *owner; /* device the glow lives on */
  cl_mem glow;     /* 3 planes */
  unsigned char key[EKV_KEY_MAX];
  size_t key_len;
  int hits;
};

typedef struct _kz_key_t
{
  uint64_t hash, mask_hash;
  int iw, ih, pad, has_mask, mask_invert;
  float diag, clip, thr, knee, hue_radius, neutral_point, softness, bias, band_split;
  ekvido_kernel_spec_t spec;
  float coef[2][3][EKVIDO_KAZUKI_NB];
} _kz_key_t;

typedef struct _mist_key_t
{
  uint64_t hash;
  int iw, ih, pad;
  float diag;
  ekvido_kernel_spec_t spec;
} _mist_key_t;

_Static_assert(sizeof(_kz_key_t) <= EKV_KEY_MAX && sizeof(_mist_key_t) <= EKV_KEY_MAX, "cache key too large");

ekv_cl_cache_t *ekv_cl_cache_new(void)
{
  return calloc(1, sizeof(ekv_cl_cache_t));
}

static void _drop(ekv_cl_cache_t *c)
{
  ekv_cl_release(c->glow);
  c->glow = NULL;
  c->owner = NULL;
  c->key_len = 0;
}

void ekv_cl_cache_free(ekv_cl_cache_t *c)
{
  if(!c) return;
  _drop(c);
  free(c);
}

int ekv_cl_cache_hits(const ekv_cl_cache_t *c)
{
  return c ? c->hits : 0;
}

static int _match(const ekv_cl_cache_t *c, const ekv_cl_t *E, const void *key, const size_t len)
{
  return c->glow && c->owner == E && c->key_len == len && !memcmp(c->key, key, len);
}

static void _store(ekv_cl_cache_t *c, ekv_cl_t *E, const void *key, const size_t len, cl_mem glow)
{
  c->owner = E;
  c->glow = glow;
  memcpy(c->key, key, len);
  c->key_len = len;
}

/* kazuki glow (3 planes, iw·ih each) from energy and warmth: band spectra, then per basis plane
   FFT → MAC into 3 accumulators, then 3 inverse FFTs. NULL on failure. */
static cl_mem _kz_glow(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem energy, cl_mem t, const ekvido_kazuki_params_t *p,
                       const float diag_px)
{
  const size_t np = (size_t)g->h * g->w, gb = ekv_cl_grid_bytes(g);
  float coef[2][3][EKVIDO_KAZUKI_NB];
  ekvido_kazuki_coefficients(p, coef);
  int r = 0;
  float *kernel = ekvido_build_kernel(&p->spec, diag_px, &r);
  if(!kernel) return NULL;
  const size_t kn = (size_t)(2 * r + 1) * (2 * r + 1);
  float *bands = malloc(sizeof(float) * kn * 2);
  if(!bands)
  {
    free(kernel);
    return NULL;
  }
  ekvido_split_bands(kernel, r, (double)p->band_split * (double)diag_px, bands, bands + kn);
  free(kernel);
  cl_mem ki = ekv_cl_buffer(E, gb), ko = ki ? ekv_cl_buffer(E, gb) : NULL, grid = ko ? ekv_cl_buffer(E, gb) : NULL;
  cl_mem acc[3] = { NULL, NULL, NULL };
  for(int c = 0; c < 3 && grid; c++) acc[c] = (c == 0 || acc[c - 1]) ? ekv_cl_buffer(E, gb) : NULL;
  int err = !grid || !acc[0] || !acc[1] || !acc[2]
            || ekv_cl_grid_kernel(E, g, bands, r, ki) || ekv_cl_grid_kernel(E, g, bands + kn, r, ko);
  free(bands);
  for(int k = 0; k < EKVIDO_KAZUKI_NB && !err; k++)
    err = ekv_cl_grid_basis(E, g, energy, t, k, grid) || ekv_cl_fft(E, g, grid, 0)
          || ekv_cl_mac(E, g, grid, ki, ko, coef, k, acc);
  ekv_cl_release(ki); /* before the inverse: lower peak */
  ekv_cl_release(ko);
  ekv_cl_release(grid);
  cl_mem glow = err ? NULL : ekv_cl_buffer(E, sizeof(float) * np * 3);
  err = err || !glow;
  for(int c = 0; c < 3 && !err; c++) err = ekv_cl_fft(E, g, acc[c], 1) || ekv_cl_crop(E, g, acc[c], glow, (size_t)c * np);
  for(int c = 0; c < 3; c++) ekv_cl_release(acc[c]);
  ekv_cl_fft_end(E); /* the FFT scratch lives only through the FFT phase */
  if(err)
  {
    ekv_cl_release(glow);
    return NULL;
  }
  return glow;
}

int ekv_cl_kazuki_process(ekv_cl_t *E, cl_mem in, const int iw, const int ih, cl_mem out, const ekvido_rect_t rect,
                          const ekvido_kazuki_params_t *p, const float diag_px, const float clip_level,
                          const float *source_mask, const int source_mask_invert, const int pad_to_cap,
                          const ekvido_kazuki_view_t view, ekv_cl_cache_t *cache)
{
  if(view == EKVIDO_VIEW_FINAL && p->strength == 0.f) return ekv_cl_copy_rect(E, in, out, rect);
  const size_t np = (size_t)iw * ih, pb = sizeof(float) * np;
  const int pad = pad_to_cap ? ekvido_kernel_half_cap(diag_px) : ekvido_kernel_half_size(&p->spec, diag_px);
  const int need_glow = view == EKVIDO_VIEW_FINAL || view == EKVIDO_VIEW_GLOW;
  int err = 0;
  _kz_key_t key;
  memset(&key, 0, sizeof(key)); /* padding takes part in memcmp */
  if(cache && need_glow)
  {
    key.hash = ekv_cl_hash_image(E, in, iw, ih, &err);
    if(err) return -2;
    key.mask_hash = source_mask ? ekvido_hash_floats(source_mask, np) : 0;
    key.iw = iw;
    key.ih = ih;
    key.pad = pad;
    key.has_mask = source_mask != NULL;
    key.mask_invert = source_mask ? (source_mask_invert != 0) : 0;
    key.diag = diag_px;
    key.clip = clip_level;
    key.thr = p->threshold_ev;
    key.knee = p->knee_ev;
    key.hue_radius = p->hue_radius;
    key.neutral_point = p->neutral_point;
    key.softness = p->warmth_softness;
    key.bias = p->neutral_bias;
    key.band_split = p->band_split;
    memcpy(&key.spec, &p->spec, sizeof(key.spec));
    ekvido_kazuki_coefficients(p, key.coef);
  }
  const int hit = cache && need_glow && _match(cache, E, &key, sizeof(key));
  if(cache && need_glow && !hit) _drop(cache); /* before the new glow is built: lower peak */
  cl_mem energy = NULL, t = NULL, clipped = NULL, glow = NULL;
  if(view != EKVIDO_VIEW_GLOW || !hit) /* a cached glow view needs no source stage at all */
  {
    cl_mem src = ekv_cl_buffer(E, 3 * pb);
    energy = src ? ekv_cl_buffer(E, 3 * pb) : NULL;
    err = !energy || ekv_cl_split(E, in, iw, ih, src) || ekv_cl_energy(E, src, np, p->threshold_ev, p->knee_ev, energy);
    if(!err && source_mask)
    {
      cl_mem m = ekv_cl_buffer(E, pb);
      err = !m || ekv_cl_upload(E, m, source_mask, pb) || ekv_cl_mask_energy(E, energy, m, np, source_mask_invert);
      ekv_cl_release(m);
    }
    if(!err && !hit) /* warmth: the glow and the warmth / clipped views */
    {
      t = ekv_cl_buffer(E, pb);
      clipped = t ? ekv_cl_buffer(E, np) : NULL;
      const float y_floor = (float)exp2((double)p->threshold_ev);
      err = !clipped
            || ekv_cl_warmth_map(E, src, ih, iw, clip_level, (double)p->hue_radius * (double)diag_px, y_floor,
                                 p->neutral_point, p->warmth_softness, p->neutral_bias, t, clipped);
    }
    ekv_cl_release(src);
  }
  if(!err && need_glow)
  {
    if(hit)
    {
      glow = cache->glow;
      cache->hits++;
    }
    else
    {
      const ekv_cl_grid_t g = ekv_cl_grid_make(ih, iw, pad);
      glow = _kz_glow(E, &g, energy, t, p, diag_px);
      err = !glow;
      if(!err && cache) _store(cache, E, &key, sizeof(key), glow);
    }
  }
  if(!err)
    err = ekv_cl_kazuki_out(E, in, out, rect, iw, ih, view, glow, energy, t, clipped, p->blend, p->strength,
                            (float)exp2((double)p->blend_white_ev));
  ekv_cl_release(energy);
  ekv_cl_release(t);
  ekv_cl_release(clipped);
  const int owned_by_cache = cache && glow && cache->glow == glow;
  if(err && owned_by_cache) _drop(cache); /* a failed run caches nothing */
  else if(!owned_by_cache) ekv_cl_release(glow);
  return err ? -2 : 0;
}

/* mist glow (3 planes, already divided by the coverage). NULL on failure. */
static cl_mem _mist_glow(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem in, const ekvido_kernel_spec_t *spec,
                         const float diag_px)
{
  const size_t np = (size_t)g->h * g->w, pb = sizeof(float) * np, gb = ekv_cl_grid_bytes(g);
  int r = 0;
  float *kernel = ekvido_build_kernel(spec, diag_px, &r);
  float *cover = kernel ? malloc(pb) : NULL;
  int err = !cover || ekvido_conv_coverage(g->h, g->w, g->pad, kernel, r, cover, 1);
  cl_mem src = err ? NULL : ekv_cl_buffer(E, 3 * pb);
  cl_mem kf = src ? ekv_cl_buffer(E, gb) : NULL, grid = kf ? ekv_cl_buffer(E, gb) : NULL;
  cl_mem glow = grid ? ekv_cl_buffer(E, 3 * pb) : NULL;
  err = err || !glow || ekv_cl_split(E, in, g->w, g->h, src) || ekv_cl_grid_kernel(E, g, kernel, r, kf);
  free(kernel);
  for(int c = 0; c < 3 && !err; c++)
    err = ekv_cl_grid_basis(E, g, src, src, c, grid) || ekv_cl_fft(E, g, grid, 0) || ekv_cl_cmul(E, g, grid, kf)
          || ekv_cl_fft(E, g, grid, 1) || ekv_cl_crop(E, g, grid, glow, (size_t)c * np);
  ekv_cl_release(src);
  ekv_cl_release(kf);
  ekv_cl_release(grid);
  ekv_cl_fft_end(E); /* the FFT scratch lives only through the FFT phase */
  cl_mem cd = err ? NULL : ekv_cl_buffer(E, pb);
  err = err || !cd || ekv_cl_upload(E, cd, cover, pb) || ekv_cl_div_cover(E, glow, cd, np);
  ekv_cl_release(cd);
  free(cover);
  if(err)
  {
    ekv_cl_release(glow);
    return NULL;
  }
  return glow;
}

int ekv_cl_mist_process(ekv_cl_t *E, cl_mem in, const int iw, const int ih, cl_mem out, const ekvido_rect_t rect,
                        const ekvido_mist_params_t *p, const float diag_px, const int pad_to_cap, const int glow_only,
                        ekv_cl_cache_t *cache)
{
  if(p->strength == 0.f && !glow_only) return ekv_cl_copy_rect(E, in, out, rect);
  const int pad = pad_to_cap ? ekvido_kernel_half_cap(diag_px) : ekvido_kernel_half_size(&p->spec, diag_px);
  int err = 0;
  _mist_key_t key;
  memset(&key, 0, sizeof(key));
  if(cache)
  {
    key.hash = ekv_cl_hash_image(E, in, iw, ih, &err);
    if(err) return -2;
    key.iw = iw;
    key.ih = ih;
    key.pad = pad;
    key.diag = diag_px;
    memcpy(&key.spec, &p->spec, sizeof(key.spec));
  }
  const int hit = cache && _match(cache, E, &key, sizeof(key));
  cl_mem glow = NULL;
  if(hit)
  {
    glow = cache->glow;
    cache->hits++;
  }
  else
  {
    if(cache) _drop(cache);
    const ekv_cl_grid_t g = ekv_cl_grid_make(ih, iw, pad);
    glow = _mist_glow(E, &g, in, &p->spec, diag_px);
    err = !glow;
    if(!err && cache) _store(cache, E, &key, sizeof(key), glow);
  }
  if(!err) err = ekv_cl_mist_out(E, in, out, rect, iw, ih, glow, p->strength, glow_only);
  const int owned_by_cache = cache && glow && cache->glow == glow;
  if(err && owned_by_cache) _drop(cache);
  else if(!owned_by_cache) ekv_cl_release(glow);
  return err ? -2 : 0;
}

static size_t _max3(const size_t a, const size_t b, const size_t c)
{
  const size_t m = a > b ? a : b;
  return m > c ? m : c;
}

size_t ekv_cl_kazuki_bytes(const int iw, const int ih, const int pad)
{
  const ekv_cl_grid_t g = ekv_cl_grid_make(ih, iw, pad);
  const size_t pl = sizeof(float) * (size_t)iw * ih, gr = ekv_cl_grid_bytes(&g);
  /* source stage: src 3 + energy 3 + t 1 + clipped ¼ + 7 warmth planes + 1 blur scratch */
  const size_t source = pl * 15 + pl / 4;
  /* spectra: energy 3 + t 1 + clipped ¼ + 2 band spectra + 1 grid + 3 accumulators (+ FFT scratch) */
  const size_t kup = sizeof(float) * (size_t)(2 * pad + 1) * (2 * pad + 1); /* grid_kernel upload, alive with the grids */
  const size_t spectra = pl * 4 + pl / 4 + gr * (6 + EKV_CL_FFT_TEMP_GRIDS) + kup;
  /* inverse: energy, t, clipped + glow 3 + 3 accumulators (+ FFT scratch) */
  const size_t inverse = pl * 7 + pl / 4 + gr * (3 + EKV_CL_FFT_TEMP_GRIDS);
  return _max3(source, spectra, inverse);
}

size_t ekv_cl_mist_bytes(const int iw, const int ih, const int pad)
{
  const ekv_cl_grid_t g = ekv_cl_grid_make(ih, iw, pad);
  const size_t pl = sizeof(float) * (size_t)iw * ih, gr = ekv_cl_grid_bytes(&g);
  const size_t kup = sizeof(float) * (size_t)(2 * pad + 1) * (2 * pad + 1); /* grid_kernel upload */
  /* src 3 + glow 3 + kernel spectrum + grid + kernel upload */
  const size_t spectra = pl * 6 + gr * (2 + EKV_CL_FFT_TEMP_GRIDS) + kup;
  const size_t cover = pl * 4;                                       /* glow 3 + coverage */
  return spectra > cover ? spectra : cover;
}
#endif /* HAVE_OPENCL */
