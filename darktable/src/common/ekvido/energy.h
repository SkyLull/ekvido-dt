#pragma once
/* Highlight energy and clip detection, mirroring ekvido/common/energy.py. Planes are
   float32 H*W, RGB as three separate planes. */
#include <math.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define EKVIDO_Y_R 0.2627f
#define EKVIDO_Y_G 0.6780f
#define EKVIDO_Y_B 0.0593f

/* NaN/±Inf → 0, negatives → 0. Every glow computation starts here. */
static inline float ekvido_positive(float v)
{
  return isfinite(v) && v > 0.f ? v : 0.f;
}

static inline float ekvido_luminance(float r, float g, float b)
{
  return r * EKVIDO_Y_R + g * EKVIDO_Y_G + b * EKVIDO_Y_B;
}

/* Python smoothstep: lo/hi are Python floats (double); the step itself runs in float32. */
static inline float ekvido_smoothstep(float x, double lo, double hi)
{
  if(hi <= lo) return x >= (float)lo ? 1.f : 0.f;
  float t = (x - (float)lo) / (float)(hi - lo);
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  return t * t * (3.f - 2.f * t);
}

/* e = src * smoothstep(log2(max(Y, 1e-8)), thr - knee, thr + knee); src already positive. */
void ekvido_extract_energy(const float *const src[3], size_t np, float threshold_ev, float knee_ev,
                           float *const energy[3]);

/* Kazuki source mask (spec §5.7): energy *= M per pixel, M = min(positive(mask), 1)
   (NaN/±Inf → 0), or 1 − M when inverted. Mirrors energy.mask_energy. */
void ekvido_mask_energy(float *const energy[3], const float *mask, size_t np, int invert);

/* Clip level from the pile-up at the top of the max-channel histogram (Python
   estimate_clip_level). `px` is interleaved with `channels` floats per pixel (RGB first);
   returns INFINITY when nothing is clipped. */
float ekvido_estimate_clip_level(const float *px, size_t npix, int channels);

#ifdef __cplusplus
}
#endif
