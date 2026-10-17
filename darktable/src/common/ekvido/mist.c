#include "common/ekvido/mist.h"
#include "common/ekvido/convolve.h"
#include "common/ekvido/hash.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct ekvido_mist_cache_t
{
  uint64_t hash;
  int iw, ih, pad;
  ekvido_conv_t *conv;
  ekvido_kernel_spec_t spec;
  float diag;
  float *glow; /* 3 planes, already divided by coverage; valid with conv's key + spec/diag */
  int glow_hits;
};

ekvido_mist_cache_t *ekvido_mist_cache_new(void)
{
  return calloc(1, sizeof(ekvido_mist_cache_t));
}

void ekvido_mist_cache_free(ekvido_mist_cache_t *c)
{
  if(!c) return;
  ekvido_conv_free(c->conv);
  free(c->glow);
  free(c);
}

int ekvido_mist_cache_glow_hits(const ekvido_mist_cache_t *c)
{
  return c ? c->glow_hits : 0;
}

static inline float _positive(float v)
{
  return isfinite(v) && v > 0.f ? v : 0.f;
}

/* Source spectra of positive(RGB) on a grid padded by `pad`. */
static ekvido_conv_t *_source(const float *in, int iw, int ih, int pad, int nthreads)
{
  const size_t np = (size_t)iw * ih;
  float *planes = malloc(sizeof(float) * np * 3);
  if(!planes) return NULL;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t i = 0; i < np; i++)
    for(int c = 0; c < 3; c++) planes[c * np + i] = _positive(in[i * 4 + c]);
  const float *pp[3] = { planes, planes + np, planes + 2 * np };
  ekvido_conv_t *conv = ekvido_conv_new(pp, 3, ih, iw, pad, nthreads);
  free(planes);
  return conv;
}

/* glow = conv(source, kernel) / max(coverage, 1e-6); returns malloc'd 3 planes or NULL. */
static float *_glow(const ekvido_conv_t *conv, int iw, int ih, const ekvido_kernel_spec_t *spec, float diag, int nthreads)
{
  const size_t np = (size_t)iw * ih;
  int r = 0;
  float *kernel = ekvido_build_kernel(spec, diag, &r);
  float *glow = malloc(sizeof(float) * np * 3);
  float *cover = malloc(sizeof(float) * np);
  int err = !kernel || !glow || !cover;
  if(!err)
  {
    float *outs[3] = { glow, glow + np, glow + 2 * np };
    err = ekvido_conv_apply(conv, kernel, r, outs)
          || ekvido_conv_coverage(ih, iw, ekvido_conv_pad(conv), kernel, r, cover, nthreads);
  }
  if(!err)
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t i = 0; i < np; i++)
    {
      const float inv = 1.f / fmaxf(cover[i], 1e-6f);
      for(int c = 0; c < 3; c++) glow[c * np + i] *= inv;
    }
  }
  free(kernel);
  free(cover);
  if(err) { free(glow); return NULL; }
  return glow;
}

int ekvido_mist_process(const float *in, int iw, int ih, float *out, ekvido_rect_t rect,
                        const ekvido_mist_params_t *p, float diag_px, int pad_to_cap, int glow_only,
                        ekvido_mist_cache_t *cache, int nthreads)
{
  const size_t np = (size_t)iw * ih;
  if(p->strength == 0.f && !glow_only)
  {
    for(int y = 0; y < rect.h; y++)
      memcpy(out + (size_t)y * rect.w * 4, in + ((size_t)(rect.y + y) * iw + rect.x) * 4, sizeof(float) * (size_t)rect.w * 4);
    return 0;
  }
  const int pad = pad_to_cap ? ekvido_kernel_half_cap(diag_px) : ekvido_kernel_half_size(&p->spec, diag_px);
  const uint64_t hash = cache ? ekvido_hash_floats(in, np * 4) : 0;
  const int same_src = cache && cache->conv && cache->hash == hash && cache->iw == iw && cache->ih == ih && cache->pad == pad;
  const int same_glow = same_src && cache->glow && !memcmp(&cache->spec, &p->spec, sizeof(p->spec)) && cache->diag == diag_px;

  ekvido_conv_t *conv = NULL;
  float *glow = NULL;
  if(same_glow)
  {
    glow = cache->glow;
    cache->glow_hits++;
  }
  else
  {
    conv = same_src ? cache->conv : _source(in, iw, ih, pad, nthreads);
    if(!conv) return -2;
    glow = _glow(conv, iw, ih, &p->spec, diag_px, nthreads);
    if(cache)
    {
      if(!same_src) { ekvido_conv_free(cache->conv); cache->conv = conv; cache->hash = hash; cache->iw = iw; cache->ih = ih; cache->pad = pad; }
      free(cache->glow);
      cache->glow = glow;
      cache->spec = p->spec;
      cache->diag = diag_px;
    }
    if(!glow)
    {
      if(!cache) ekvido_conv_free(conv);
      return -2;
    }
  }

  const float k = p->strength;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int y = 0; y < rect.h; y++)
    for(int x = 0; x < rect.w; x++)
    {
      const size_t si = (size_t)(rect.y + y) * iw + (rect.x + x);
      float *o = out + ((size_t)y * rect.w + x) * 4;
      const float *s = in + si * 4;
      for(int c = 0; c < 3; c++)
      {
        const float g = glow[c * np + si];
        /* Non-finite inputs written out explicitly: -ffast-math (darktable's release flags) may
           reassociate the mix into s + k(g - s), turning +Inf into NaN. Python: ±Inf stays ±Inf
           for k < 1 and 0·Inf = NaN at k = 1; NaN stays NaN. */
        o[c] = glow_only ? g : isfinite(s[c]) ? (1.f - k) * s[c] + k * g : (k < 1.f ? s[c] : NAN);
      }
      o[3] = s[3];
    }
  if(!cache) { ekvido_conv_free(conv); free(glow); }
  return 0;
}
