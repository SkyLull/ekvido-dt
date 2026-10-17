#include "common/ekvido/kazuki.h"
#include "common/ekvido/color.h"
#include "common/ekvido/convolve.h"
#include "common/ekvido/energy.h"
#include "common/ekvido/hash.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NB EKVIDO_KAZUKI_NB

#if defined(__GNUC__) || defined(__clang__)
#define EKV_NOINLINE __attribute__((noinline))
#else
#define EKV_NOINLINE
#endif

typedef struct _src_key_t
{
  uint64_t hash;
  int iw, ih, pad;
  float diag, clip, thr, knee, hue_radius, neutral_point, softness, bias;
  uint64_t mask_hash; /* 0 and has_mask 0 without a source mask */
  int has_mask, mask_invert;
} _src_key_t;

struct ekvido_kazuki_cache_t
{
  _src_key_t key;
  int src_valid;
  float *energy;          /* 3 planes */
  float *t;               /* warmth */
  unsigned char *clipped;
  float *spectra;         /* NB basis spectra on the key's grid; built on the first glow need */
  ekvido_kernel_spec_t spec;
  float band_split;
  float *kspec;           /* inner + outer band spectra; valid with key + spec + band_split */
  float coef[2][3][NB];
  float *glow;            /* 3 planes; holds a glow when glow_valid (with kspec + coef) */
  int glow_valid;
  float *acc, *work;      /* scratch on the key's grid, kept between calls: no fresh pages per slider move */
  int src_hits, kernel_hits, glow_hits;
};

ekvido_kazuki_cache_t *ekvido_kazuki_cache_new(void)
{
  return calloc(1, sizeof(ekvido_kazuki_cache_t));
}

/* The glow buffer stays allocated (same grid until the source changes); only its contents go. */
static void _drop_glow(ekvido_kazuki_cache_t *c)
{
  c->glow_valid = 0;
}

static void _drop_kernel(ekvido_kazuki_cache_t *c)
{
  _drop_glow(c);
  free(c->kspec);
  c->kspec = NULL;
}

static void _drop_source(ekvido_kazuki_cache_t *c)
{
  _drop_kernel(c);
  free(c->energy); free(c->t); free(c->clipped); free(c->spectra);
  free(c->glow); free(c->acc); free(c->work);
  c->energy = c->t = c->spectra = c->glow = c->acc = c->work = NULL;
  c->clipped = NULL;
  c->src_valid = 0;
}

void ekvido_kazuki_cache_free(ekvido_kazuki_cache_t *c)
{
  if(!c) return;
  _drop_source(c);
  free(c);
}

void ekvido_kazuki_cache_hits(const ekvido_kazuki_cache_t *c, int *source_hits, int *kernel_hits, int *glow_hits)
{
  *source_hits = c ? c->src_hits : 0;
  *kernel_hits = c ? c->kernel_hits : 0;
  *glow_hits = c ? c->glow_hits : 0;
}

void ekvido_kazuki_basis_plane(const float *const e[3], const float *t, size_t np, int k, float *out)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t i = 0; i < np; i++)
  {
    if(k < 3)
      out[i] = e[k][i];
    else if(k < 6)
      out[i] = t[i] * e[k - 3][i];
    else
      out[i] = t[i] * t[i] * ekvido_luminance(e[0][i], e[1][i], e[2][i]);
  }
}

void ekvido_kazuki_coefficients(const ekvido_kazuki_params_t *p, float coef[2][3][NB])
{
  const float wy[3] = { EKVIDO_Y_R, EKVIDO_Y_G, EKVIDO_Y_B };
  const float pc = p->push_cool, dp = (float)((double)p->push_warm - (double)p->push_cool);
  for(int band = 0; band < 2; band++)
  {
    float cu[3], wu[3], du[3];
    ekvido_unit_luminance(band ? p->cool_edge : p->cool_core, cu);
    ekvido_unit_luminance(band ? p->warm_edge : p->warm_core, wu);
    for(int c = 0; c < 3; c++) du[c] = wu[c] - cu[c];
    for(int c = 0; c < 3; c++)
    {
      /* weights of Python's planes Y(e), t·Y(e), t²·Y(e) for output c */
      const float a6 = cu[c] * pc, a7 = cu[c] * dp + du[c] * pc, a8 = du[c] * dp;
      for(int j = 0; j < 3; j++)
      {
        coef[band][c][j] = (j == c ? 1.f - pc : 0.f) + wy[j] * a6;
        coef[band][c][3 + j] = (j == c ? -dp : 0.f) + wy[j] * a7;
      }
      coef[band][c][6] = a8;
    }
  }
}

int ekvido_kazuki_margin(const ekvido_kazuki_params_t *p, float diag_px)
{
  const double sigma = (double)p->hue_radius * (double)diag_px;
  int blur = (int)ceil(3.0 * sigma);
  if(sigma > 8.0) blur += 2 * (int)floor(sigma / 4.0); /* downsampled blur: block + upsample reach */
  const int m = ekvido_kernel_half_size(&p->spec, diag_px) + blur;
  const int cap = ekvido_kernel_half_cap(diag_px);
  return m < cap ? m : cap;
}

/* Source stage: energy, warmth and the clipped mask. 0 or -2. */
static int _source(const float *in, int iw, int ih, const ekvido_kazuki_params_t *p, float diag_px, float clip_level,
                   const float *mask, int mask_invert, float **energy, float **t, unsigned char **clipped)
{
  const size_t np = (size_t)iw * ih;
  float *src = malloc(sizeof(float) * np * 3);
  *energy = malloc(sizeof(float) * np * 3);
  *t = malloc(sizeof(float) * np);
  *clipped = malloc(np);
  int err = !src || !*energy || !*t || !*clipped;
  if(!err)
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t i = 0; i < np; i++)
      for(int c = 0; c < 3; c++) src[c * np + i] = ekvido_positive(in[i * 4 + c]);
    const float *s3[3] = { src, src + np, src + 2 * np };
    float *e3[3] = { *energy, *energy + np, *energy + 2 * np };
    ekvido_extract_energy(s3, np, p->threshold_ev, p->knee_ev, e3);
    if(mask) ekvido_mask_energy(e3, mask, np, mask_invert); /* warmth below still sees everything */
    const float y_floor = (float)exp2((double)p->threshold_ev); /* threshold luminance */
    err = ekvido_warmth_map(s3, ih, iw, clip_level, (double)p->hue_radius * (double)diag_px, y_floor,
                            p->neutral_point, p->warmth_softness, p->neutral_bias, *t, *clipped);
  }
  free(src);
  if(err)
  {
    free(*energy); free(*t); free(*clipped);
    *energy = *t = NULL;
    *clipped = NULL;
    return -2;
  }
  return 0;
}

/* Inner + outer band spectra of the glow kernel on grid g (2 half-spectra, malloc'd), or NULL. */
static float *_band_spectra(const ekvido_grid_t *g, const ekvido_kazuki_params_t *p, float diag_px, float *work)
{
  int r = 0;
  float *kernel = ekvido_build_kernel(&p->spec, diag_px, &r);
  if(!kernel) return NULL;
  const size_t kn = (size_t)(2 * r + 1) * (2 * r + 1), sf = ekvido_grid_spec_floats(g);
  float *bands = malloc(sizeof(float) * kn * 2);
  float *ks = bands ? malloc(sizeof(float) * sf * 2) : NULL;
  if(!ks) { free(kernel); free(bands); return NULL; }
  ekvido_split_bands(kernel, r, (double)p->band_split * (double)diag_px, bands, bands + kn);
  free(kernel);
  const int err = ekvido_grid_kernel_spectrum(g, bands, r, work, ks)
                  || ekvido_grid_kernel_spectrum(g, bands + kn, r, work, ks + sf);
  free(bands);
  if(err) { free(ks); return NULL; }
  return ks;
}

/* The NB basis spectra (malloc'd, NB half-spectra), or NULL. */
static float *_basis_spectra(const ekvido_grid_t *g, const float *energy, const float *t, float *work)
{
  const size_t np = (size_t)g->h * g->w, sf = ekvido_grid_spec_floats(g);
  float *spectra = malloc(sizeof(float) * sf * NB), *plane = malloc(sizeof(float) * np);
  if(!spectra || !plane) { free(spectra); free(plane); return NULL; }
  const float *e3[3] = { energy, energy + np, energy + 2 * np };
  for(int k = 0; k < NB; k++)
  {
    ekvido_kazuki_basis_plane(e3, t, np, k, plane);
    ekvido_grid_plane_spectrum(g, plane, work, spectra + (size_t)k * sf);
  }
  free(plane);
  return spectra;
}

#define MAC_TILE 4096 /* complex bins per tile: acc tiles stay in cache across the NB planes */

/* For bins [m0, m1): acc_c (=, k == 0 / +=) S·(coef[0][c][k]·Ki + coef[1][c][k]·Ko), acc = 3
   half-spectra sf floats apart. The one place glow spectra are summed: every bin of the cached and
   the streaming path goes through this one serial, not-inlined function with the same operands in
   plane order, so their results are bit-identical whatever the tiling or thread split, and
   -ffast-math cannot contract two call sites differently. coef is read only (not const: C warns on
   passing float[2][3][NB] to a const 2-D array parameter). */
static EKV_NOINLINE void _mac_tile(const float *s, const float *ki, const float *ko, float coef[2][3][NB], int k,
                                   float *acc, size_t sf, size_t m0, size_t m1)
{
  const float ci[3] = { coef[0][0][k], coef[0][1][k], coef[0][2][k] };
  const float co[3] = { coef[1][0][k], coef[1][1][k], coef[1][2][k] };
  for(size_t m = m0; m < m1; m++)
  {
    const size_t q = 2 * m;
    const float a = s[q], b = s[q + 1];
    const float ir = a * ki[q] - b * ki[q + 1], ii = a * ki[q + 1] + b * ki[q];
    const float orr = a * ko[q] - b * ko[q + 1], oi = a * ko[q + 1] + b * ko[q];
    for(int c = 0; c < 3; c++)
    {
      float *ac = acc + (size_t)c * sf;
      const float re = ci[c] * ir + co[c] * orr, im = ci[c] * ii + co[c] * oi;
      ac[q] = k ? ac[q] + re : re;
      ac[q + 1] = k ? ac[q + 1] + im : im;
    }
  }
}

/* Planes k0…k1-1 (spectra s, s + sf, …) into acc, tile by tile: all planes of a tile before the
   next tile, so the cached path (k0 = 0, k1 = NB) streams each spectrum once instead of re-reading
   and re-writing acc NB times. The streaming path calls it one plane at a time. */
static void _mac(const float *s, int k0, int k1, const float *ki, const float *ko, float coef[2][3][NB],
                 float *acc, size_t sf)
{
  const size_t nc = sf / 2, ntiles = (nc + MAC_TILE - 1) / MAC_TILE;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t tile = 0; tile < ntiles; tile++)
  {
    const size_t m0 = tile * MAC_TILE, m1 = m0 + MAC_TILE < nc ? m0 + MAC_TILE : nc;
    for(int k = k0; k < k1; k++) _mac_tile(s + (size_t)(k - k0) * sf, ki, ko, coef, k, acc, sf, m0, m1);
  }
}

/* glow (3 planes) = inverse of the 3 accumulated spectra. */
static void _inverse3(const ekvido_grid_t *g, const float *acc, float *work, float *glow)
{
  const size_t np = (size_t)g->h * g->w, sf = ekvido_grid_spec_floats(g);
  for(int c = 0; c < 3; c++) ekvido_grid_inverse(g, acc + (size_t)c * sf, work, glow + (size_t)c * np);
}

/* Export path (no cache): band spectra first, then each basis plane is transformed, accumulated
   and dropped — one plane spectrum alive at a time. */
static float *_glow_stream(const ekvido_grid_t *g, const float *energy, const float *t, const ekvido_kazuki_params_t *p,
                           float diag_px)
{
  const size_t np = (size_t)g->h * g->w, sf = ekvido_grid_spec_floats(g);
  float coef[2][3][NB];
  ekvido_kazuki_coefficients(p, coef);
  float *work = malloc(sizeof(float) * ekvido_grid_floats(g));
  float *ks = work ? _band_spectra(g, p, diag_px, work) : NULL;
  float *plane = ks ? malloc(sizeof(float) * np) : NULL;
  float *spec = plane ? malloc(sizeof(float) * sf) : NULL;
  float *acc = spec ? malloc(sizeof(float) * sf * 3) : NULL;
  float *glow = NULL;
  if(acc)
  {
    const float *e3[3] = { energy, energy + np, energy + 2 * np };
    for(int k = 0; k < NB; k++)
    {
      ekvido_kazuki_basis_plane(e3, t, np, k, plane);
      ekvido_grid_plane_spectrum(g, plane, work, spec);
      _mac(spec, k, k + 1, ks, ks + sf, coef, acc, sf);
    }
    free(plane); plane = NULL; /* drop before the inverse: lower peak */
    free(spec); spec = NULL;
    free(ks); ks = NULL;
    glow = malloc(sizeof(float) * np * 3);
    if(glow) _inverse3(g, acc, work, glow);
  }
  free(work); free(ks); free(plane); free(spec); free(acc);
  return glow;
}

/* Interactive path: a strength/blend/white change reuses the glow; a colour/pull change reuses
   both spectra sets (one accumulate pass + 3 inverse FFTs); a shape change reuses the source
   spectra. The glow, acc and work buffers stay owned by the cache, so a slider move allocates
   nothing on the grid. */
static const float *_glow_cached(const ekvido_grid_t *g, ekvido_kazuki_cache_t *c, const ekvido_kazuki_params_t *p,
                                 float diag_px)
{
  float coef[2][3][NB];
  ekvido_kazuki_coefficients(p, coef);
  const int same_kernel = c->kspec && !memcmp(&c->spec, &p->spec, sizeof(p->spec)) && c->band_split == p->band_split;
  if(same_kernel && c->glow_valid && !memcmp(c->coef, coef, sizeof(coef)))
  {
    c->glow_hits++;
    return c->glow;
  }
  _drop_glow(c);
  if(!same_kernel) _drop_kernel(c); /* free stale spectra before building new ones: peak memory */
  const size_t np = (size_t)g->h * g->w, sf = ekvido_grid_spec_floats(g);
  if(!c->work) c->work = malloc(sizeof(float) * ekvido_grid_floats(g));
  if(!c->work) return NULL;
  if(!c->spectra) c->spectra = _basis_spectra(g, c->energy, c->t, c->work);
  if(c->spectra && same_kernel)
    c->kernel_hits++;
  else if(c->spectra)
  {
    c->kspec = _band_spectra(g, p, diag_px, c->work);
    if(c->kspec)
    {
      c->spec = p->spec;
      c->band_split = p->band_split;
    }
  }
  if(!c->spectra || !c->kspec) return NULL;
  if(!c->acc) c->acc = malloc(sizeof(float) * sf * 3);
  if(!c->glow) c->glow = malloc(sizeof(float) * np * 3);
  if(!c->acc || !c->glow) return NULL;
  _mac(c->spectra, 0, NB, c->kspec, c->kspec + sf, coef, c->acc, sf);
  _inverse3(g, c->acc, c->work, c->glow);
  memcpy(c->coef, coef, sizeof(coef));
  c->glow_valid = 1;
  return c->glow;
}

int ekvido_kazuki_process(const float *in, int iw, int ih, float *out, ekvido_rect_t rect,
                          const ekvido_kazuki_params_t *p, float diag_px, float clip_level, int pad_to_cap,
                          ekvido_kazuki_view_t view, ekvido_kazuki_cache_t *cache, int nthreads)
{
  return ekvido_kazuki_process_masked(in, iw, ih, out, rect, p, diag_px, clip_level, NULL, 0, pad_to_cap, view,
                                      cache, nthreads);
}

int ekvido_kazuki_process_masked(const float *in, int iw, int ih, float *out, ekvido_rect_t rect,
                                 const ekvido_kazuki_params_t *p, float diag_px, float clip_level,
                                 const float *source_mask, int source_mask_invert, int pad_to_cap,
                                 ekvido_kazuki_view_t view, ekvido_kazuki_cache_t *cache, int nthreads)
{
  const size_t np = (size_t)iw * ih;
  if(view == EKVIDO_VIEW_FINAL && p->strength == 0.f)
  {
    for(int y = 0; y < rect.h; y++)
      memcpy(out + (size_t)y * rect.w * 4, in + ((size_t)(rect.y + y) * iw + rect.x) * 4, sizeof(float) * (size_t)rect.w * 4);
    return 0;
  }
  const int pad = pad_to_cap ? ekvido_kernel_half_cap(diag_px) : ekvido_kernel_half_size(&p->spec, diag_px);
  _src_key_t key;
  memset(&key, 0, sizeof(key)); /* padding bytes take part in memcmp */
  key.hash = cache ? ekvido_hash_floats(in, np * 4) : 0;
  key.iw = iw; key.ih = ih; key.pad = pad;
  key.diag = diag_px; key.clip = clip_level; key.thr = p->threshold_ev; key.knee = p->knee_ev;
  key.hue_radius = p->hue_radius; key.neutral_point = p->neutral_point;
  key.softness = p->warmth_softness; key.bias = p->neutral_bias;
  key.has_mask = source_mask != NULL;
  key.mask_invert = source_mask ? (source_mask_invert != 0) : 0;
  key.mask_hash = cache && source_mask ? ekvido_hash_floats(source_mask, np) : 0;

  float *energy = NULL, *t = NULL;
  unsigned char *clipped = NULL;
  if(cache && cache->src_valid && !memcmp(&cache->key, &key, sizeof(key)))
  {
    energy = cache->energy; t = cache->t; clipped = cache->clipped;
    cache->src_hits++;
  }
  else
  {
    if(cache) _drop_source(cache);
    if(_source(in, iw, ih, p, diag_px, clip_level, source_mask, source_mask_invert, &energy, &t, &clipped)) return -2;
    if(cache)
    {
      cache->key = key; cache->src_valid = 1;
      cache->energy = energy; cache->t = t; cache->clipped = clipped;
    }
  }

  const ekvido_grid_t g = ekvido_grid_make(ih, iw, pad, nthreads);
  const float *glow = NULL;
  float *own_glow = NULL;
  const int need_glow = view == EKVIDO_VIEW_FINAL || view == EKVIDO_VIEW_GLOW;
  int err = 0;
  if(need_glow)
  {
    if(cache)
      glow = _glow_cached(&g, cache, p, diag_px);
    else
      glow = own_glow = _glow_stream(&g, energy, t, p, diag_px);
    if(!glow) err = -2;
  }

  if(!err)
  {
    const float white = (float)exp2((double)p->blend_white_ev);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int y = 0; y < rect.h; y++)
      for(int x = 0; x < rect.w; x++)
      {
        const size_t si = (size_t)(rect.y + y) * iw + (rect.x + x);
        const float *s = in + si * 4;
        float *o = out + ((size_t)y * rect.w + x) * 4;
        switch(view)
        {
          case EKVIDO_VIEW_GLOW:
            for(int c = 0; c < 3; c++) o[c] = glow[c * np + si];
            break;
          case EKVIDO_VIEW_ENERGY:
            for(int c = 0; c < 3; c++) o[c] = energy[c * np + si];
            break;
          case EKVIDO_VIEW_WARMTH:
            ekvido_warmth_colour(t[si], o);
            break;
          case EKVIDO_VIEW_CLIPPED:
            for(int c = 0; c < 3; c++) o[c] = clipped[si] ? (c == 0 ? 1.f : 0.f) : s[c];
            break;
          default:
          {
            const float gl[3] = { glow[si], glow[np + si], glow[2 * np + si] };
            const float e[3] = { energy[si], energy[np + si], energy[2 * np + si] };
            ekvido_blend_px(p->blend, s, gl, e, p->strength, white, o);
            break;
          }
        }
        o[3] = s[3];
      }
  }
  free(own_glow);
  if(!cache)
  {
    free(energy); free(t); free(clipped);
  }
  return err;
}
