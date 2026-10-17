#include "common/ekvido/color.h"
#include "common/ekvido/blend.h"
#include "common/ekvido/energy.h"
#include "common/ekvido/wblur.h"

#include <math.h>
#include <stdlib.h>

void ekvido_hsv_to_rgb(float hue_deg, float sat, float rgb[3])
{
  const double h = fmod((double)hue_deg, 360.0) / 360.0, s = sat, v = 1.0;
  if(s == 0.0) { rgb[0] = rgb[1] = rgb[2] = (float)v; return; }
  int i = (int)(h * 6.0);
  const double f = h * 6.0 - i;
  const double p = v * (1.0 - s), q = v * (1.0 - s * f), t = v * (1.0 - s * (1.0 - f));
  double r, g, b;
  switch(i % 6)
  {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  rgb[0] = (float)r; rgb[1] = (float)g; rgb[2] = (float)b;
}

void ekvido_unit_luminance(const float rgb[3], float out[3])
{
  const float y = fmaxf(ekvido_luminance(rgb[0], rgb[1], rgb[2]), 1e-8f);
  for(int c = 0; c < 3; c++) out[c] = rgb[c] / y;
}

int ekvido_warmth_map(const float *const src[3], int h, int w, float clip_level, double sigma_px,
                      float y_floor, float neutral_point, float softness, float bias, float *t,
                      unsigned char *clipped)
{
  const size_t np = (size_t)h * w;
  float *buf = malloc(sizeof(float) * np * 7);
  if(!buf) return -2;
  float *wt = buf, *xr = buf + np, *xb = buf + 2 * np, *cm = buf + 3 * np;
  float *hr = buf + 4 * np, *hb = buf + 5 * np, *den = buf + 6 * np;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t i = 0; i < np; i++)
  {
    const float r = src[0][i], g = src[1][i], b = src[2][i];
    const int cl = fmaxf(fmaxf(r, g), b) >= clip_level;
    const float s = r + g + b;
    clipped[i] = (unsigned char)cl;
    wt[i] = cl ? 0.f : ekvido_luminance(r, g, b);
    xr[i] = s > 1e-12f ? r / fmaxf(s, 1e-12f) : 1.f / 3.f; /* chromaticity r, b */
    xb[i] = s > 1e-12f ? b / fmaxf(s, 1e-12f) : 1.f / 3.f;
    cm[i] = cl ? 1.f : 0.f;
  }
  const float *x[2] = { xr, xb };
  float *hue[2] = { hr, hb };
  const int err = ekvido_weighted_blur(x, 2, wt, h, w, sigma_px, hue, den)
                  || ekvido_gaussian_blur(cm, cm, h, w, sigma_px);
  if(!err)
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t i = 0; i < np; i++)
    {
      /* Confidence = unclipped light vs clipped area, each clipped pixel counted at the
         threshold luminance (not at its unknown, possibly huge, clipped value). */
      const float conf = den[i] / (den[i] + cm[i] * y_floor + 1e-12f);
      const float a = ekvido_smoothstep(conf, 0.15, 0.4);
      const float warm = ekvido_smoothstep(hr[i] - hb[i] - neutral_point, -(double)softness, (double)softness);
      t[i] = a * warm + (1.f - a) * bias;
    }
  }
  free(buf);
  return err ? -2 : 0;
}

void ekvido_warmth_colour(float t, float rgb[3])
{
  static const float cool[3] = { 40.f, 90.f, 255.f }, mid[3] = { 128.f, 128.f, 128.f },
                     warm[3] = { 255.f, 140.f, 30.f };
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  for(int c = 0; c < 3; c++)
  {
    const float v = t <= 0.5f ? cool[c] + (mid[c] - cool[c]) * (t * 2.f) : mid[c] + (warm[c] - mid[c]) * (t * 2.f - 1.f);
    rgb[c] = ekvido_srgb_decode(nearbyintf(v) / 255.f);
  }
}
