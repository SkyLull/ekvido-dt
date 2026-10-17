#include "common/ekvido/blend.h"

#include <math.h>

static inline float _clamp01(float x)
{
  return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); /* NaN stays NaN, as torch.clamp */
}

float ekvido_srgb_encode(float x)
{
  x = _clamp01(x);
  return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.f / 2.4f) - 0.055f;
}

float ekvido_srgb_decode(float v)
{
  v = _clamp01(v);
  return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}

float ekvido_dodge(float a, float c)
{
  if(a <= 0.f) return 0.f;
  if(c >= 1.f) return 1.f;
  /* NaN propagates, as torch.minimum/where do; spelled out because GCC's fast-math fminf/fmaxf
     can drop it (isnan survives -fno-finite-math-only) */
  if(isnan(a) || isnan(c)) return NAN;
  return fminf(1.f, a / fmaxf(1.f - c, 1e-12f));
}

static inline float _overlay(float a, float c)
{
  return a <= 0.5f ? 2.f * a * c : 1.f - 2.f * (1.f - a) * (1.f - c);
}

float ekvido_overlay_glow(float a, float c, float m, float alpha)
{
  if(!(m > EKVIDO_GLOW_EPS)) return a;
  return a + alpha * m * (_overlay(a, fminf(c / m, 1.f)) - a);
}

float ekvido_csp_blend(ekvido_blend_mode_t mode, float a, float c, float alpha)
{
  switch(mode)
  {
    case EKVIDO_BLEND_SCREEN:
      return a + alpha * ((1.f - (1.f - a) * (1.f - c)) - a);
    case EKVIDO_BLEND_OVERLAY:
      return a + alpha * (_overlay(a, c) - a);
    case EKVIDO_BLEND_COLOR_DODGE:
      return a + alpha * (ekvido_dodge(a, c) - a);
    case EKVIDO_BLEND_GLOW_DODGE:
      return ekvido_dodge(a, alpha * c);
    case EKVIDO_BLEND_OVERLAY_GLOW:
      return ekvido_overlay_glow(a, c, c, alpha);
    default:
      return a;
  }
}

/* One channel of a CSP-style mode: encode, blend on display values, decode; light above white
   and negative values pass through. m is the layer opacity for OVERLAY_GLOW (ignored otherwise). */
static float _blend_encoded(ekvido_blend_mode_t mode, float img, float big_c, float m, float alpha, float white)
{
  const float a = img / white;
  const float a_c = _clamp01(a);
  const float excess = a - a_c;
  const float big_a = ekvido_srgb_encode(a_c);
  const float r = mode == EKVIDO_BLEND_OVERLAY_GLOW ? ekvido_overlay_glow(big_a, big_c, m, alpha)
                                                    : ekvido_csp_blend(mode, big_a, big_c, alpha);
  return white * (ekvido_srgb_decode(r) + excess);
}

float ekvido_blend(ekvido_blend_mode_t mode, float img, float glow, float energy, float strength, float white)
{
  if(mode == EKVIDO_BLEND_ADDITIVE) return img + strength * glow;
  const float alpha = fminf(strength, 1.f);
  if(mode == EKVIDO_BLEND_CONSERVE) return img + alpha * (glow - energy);
  if(alpha == 0.f || mode == EKVIDO_BLEND_OVERLAY_GLOW) return img;
  return _blend_encoded(mode, img, ekvido_srgb_encode(fminf(glow / white, 1.f)), 0.f, alpha, white);
}

void ekvido_blend_px(ekvido_blend_mode_t mode, const float img[3], const float glow[3], const float energy[3],
                     float strength, float white, float out[3])
{
  if(mode != EKVIDO_BLEND_OVERLAY_GLOW)
  {
    for(int c = 0; c < 3; c++) out[c] = ekvido_blend(mode, img[c], glow[c], energy[c], strength, white);
    return;
  }
  const float alpha = fminf(strength, 1.f);
  float big_c[3], m = 0.f;
  for(int c = 0; c < 3; c++)
  {
    big_c[c] = ekvido_srgb_encode(fminf(glow[c] / white, 1.f));
    m = fmaxf(m, big_c[c]);
  }
  if(alpha == 0.f || !(m > EKVIDO_GLOW_EPS)) /* glow-free pixels stay exactly as they were */
  {
    for(int c = 0; c < 3; c++) out[c] = img[c];
    return;
  }
  for(int c = 0; c < 3; c++) out[c] = _blend_encoded(mode, img[c], big_c[c], m, alpha, white);
}
