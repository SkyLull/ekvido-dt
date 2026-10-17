#include "common/ekvido/energy.h"

#include <math.h>

void ekvido_extract_energy(const float *const src[3], size_t np, float threshold_ev, float knee_ev,
                           float *const energy[3])
{
  const double lo = (double)threshold_ev - knee_ev, hi = (double)threshold_ev + knee_ev;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t i = 0; i < np; i++)
  {
    const float y = ekvido_luminance(src[0][i], src[1][i], src[2][i]);
    const float k = ekvido_smoothstep(log2f(fmaxf(y, 1e-8f)), lo, hi);
    for(int c = 0; c < 3; c++) energy[c][i] = src[c][i] * k;
  }
}

void ekvido_mask_energy(float *const energy[3], const float *mask, size_t np, int invert)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t i = 0; i < np; i++)
  {
    float m = fminf(ekvido_positive(mask[i]), 1.f);
    if(invert) m = 1.f - m;
    for(int c = 0; c < 3; c++) energy[c][i] *= m;
  }
}

static inline float _max3(const float *p)
{
  return fmaxf(fmaxf(ekvido_positive(p[0]), ekvido_positive(p[1])), ekvido_positive(p[2]));
}

float ekvido_estimate_clip_level(const float *px, size_t npix, int channels)
{
  if(npix == 0) return INFINITY;
  float top = 0.f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max : top)
#endif
  for(size_t i = 0; i < npix; i++) top = fmaxf(top, _max3(px + i * channels));
  if(!(top > 0.f)) return INFINITY;
  /* A real clip is a plateau: clipped pixels share (almost) exactly the max value. */
  const float near = (float)((double)top * (1.0 - 1e-5));
  size_t count = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : count)
#endif
  for(size_t i = 0; i < npix; i++) count += _max3(px + i * channels) >= near;
  if((double)count / (double)npix > 1e-4) return (float)(0.999 * (double)top);
  return INFINITY;
}
