#include "common/ekvido/psf.h"
#include "common/ekvido/energy.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EKVIDO_FLOOR 1e-6 /* parts are cut where they fall below this fraction of their peak */

static double _moffat_extent(double r0, double beta)
{
  return r0 * sqrt(pow(EKVIDO_FLOOR, -1.0 / beta) - 1.0);
}

int ekvido_kernel_half_cap(float diag_px)
{
  const int c = (int)ceil(EKVIDO_MAX_KERNEL_FRAC * (double)diag_px);
  return c > 1 ? c : 1;
}

double ekvido_kernel_extent_frac(const ekvido_kernel_spec_t *s)
{
  double ext = 0.0;
  if(s->w_halo > 0.f)
    ext = fmax(ext, _moffat_extent(s->halo_radius, s->halo_beta) * fmax(s->halo_stretch_x, s->halo_stretch_y));
  if(s->w_veil > 0.f)
    ext = fmax(ext, _moffat_extent(s->veil_radius, s->veil_beta));
  if(s->w_arms > 0.f && s->arms > 0)
    ext = fmax(ext, _moffat_extent(s->arm_radius, s->arm_beta) * fmax(s->arm_stretch_x, s->arm_stretch_y));
  return ext;
}

int ekvido_kernel_half_size(const ekvido_kernel_spec_t *s, float diag_px)
{
  const int want = (int)ceil(ekvido_kernel_extent_frac(s) * (double)diag_px);
  const int cap = ekvido_kernel_half_cap(diag_px);
  const int r = want > 1 ? want : 1;
  return r < cap ? r : cap;
}

static inline float _smoothstep(float x, float lo, float hi)
{
  if(hi <= lo) return x >= lo ? 1.f : 0.f;
  float t = (x - lo) / (hi - lo);
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  return t * t * (3.f - 2.f * t);
}

/* Normalise a d×d kernel to sum 1. Row sums in double, added in row order: the same bytes for
   any thread count. 0, or -1 when out of memory (no fallback order: it would change the bytes). */
static int _normalise(float *k, int d)
{
  double *rows = malloc(sizeof(double) * (size_t)d);
  if(!rows) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < d; i++)
  {
    double rs = 0.0;
    for(int j = 0; j < d; j++) rs += k[(size_t)i * d + j];
    rows[i] = rs;
  }
  double s = 0.0;
  for(int i = 0; i < d; i++) s += rows[i];
  free(rows);
  if(s > 0.0)
  {
    const float inv = (float)(1.0 / s);
    const size_t n = (size_t)d * d;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t i = 0; i < n; i++) k[i] *= inv;
  }
  return 0;
}

/* out += w · part */
static void _add_part(float *out, const float *part, float w, size_t n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t q = 0; q < n; q++) out[q] += w * part[q];
}

static inline float _moffat(float x, float y, float r0_px, float beta, float sx, float sy)
{
  const float r0 = fmaxf(r0_px, 0.5f); /* below half a pixel the halo is a point anyway */
  const float u = x / sx, v = y / sy;
  return powf(1.f + (u * u + v * v) / (r0 * r0), -beta);
}

/* Moffat over the L_p distance (|u|^p + |v|^p)^(1/p), angle folded by n/4 (exact L_p for
   n = 4), then a half-pixel blur so needle-thin arms do not alias. Returns 0 or -1. */
static int _star(float *k, int r, const ekvido_kernel_spec_t *s, float r0_px)
{
  const int d = 2 * r + 1;
  const float r0 = fmaxf(r0_px, 0.5f), p = fmaxf(s->arm_sharpness, 0.05f);
  const double th = s->arm_angle * M_PI / 180.0;
  const float c = (float)cos(th), sn = (float)sin(th), fold_k = (float)s->arms / 4.f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < d; i++)
    for(int j = 0; j < d; j++)
    {
      const float x = (float)(j - r), y = (float)(i - r);
      const float xs = (x * c + y * sn) / s->arm_stretch_x, ys = (-x * sn + y * c) / s->arm_stretch_y;
      const float rad = sqrtf(xs * xs + ys * ys) / r0;
      /* atan2 in double, rounded once: libm atan2f is not correctly rounded on every platform, and
         with small sharpness the sin/cos of an on-axis fold angle amplify a 1-ulp difference. */
      const float fold = (float)atan2((double)ys, (double)xs) * fold_k;
      const float sum_p = powf(rad * fabsf(cosf(fold)), p) + powf(rad * fabsf(sinf(fold)), p);
      k[(size_t)i * d + j] = powf(1.f + powf(sum_p, 2.f / p), -s->arm_beta);
    }
  float g[5], gs = 0.f;
  for(int t = 0; t < 5; t++)
  {
    const float u = (float)(t - 2) / 0.5f;
    g[t] = expf(-0.5f * u * u);
    gs += g[t];
  }
  for(int t = 0; t < 5; t++) g[t] /= gs;
  float *tmp = malloc(sizeof(float) * (size_t)d * d);
  if(!tmp) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < d; i++) /* horizontal, zero padded */
    for(int j = 0; j < d; j++)
    {
      float acc = 0.f;
      for(int t = 0; t < 5; t++)
      {
        const int jj = j + t - 2;
        if(jj >= 0 && jj < d) acc += g[t] * k[(size_t)i * d + jj];
      }
      tmp[(size_t)i * d + j] = acc;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < d; i++) /* vertical, zero padded */
    for(int j = 0; j < d; j++)
    {
      float acc = 0.f;
      for(int t = 0; t < 5; t++)
      {
        const int ii = i + t - 2;
        if(ii >= 0 && ii < d) acc += g[t] * tmp[(size_t)ii * d + j];
      }
      k[(size_t)i * d + j] = acc;
    }
  free(tmp);
  return 0;
}

float *ekvido_build_kernel(const ekvido_kernel_spec_t *s, float diag_px, int *radius)
{
  const int r = ekvido_kernel_half_size(s, diag_px);
  const int d = 2 * r + 1;
  const size_t n = (size_t)d * d;
  float *out = calloc(n, sizeof(float));
  float *part = malloc(sizeof(float) * n);
  float *taper = malloc(sizeof(float) * n);
  if(!out || !part || !taper) { free(out); free(part); free(taper); return NULL; }
  *radius = r;
  /* Fade every part to zero over the outer 20 % of the support, inside a disc. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < d; i++)
    for(int j = 0; j < d; j++)
    {
      const float x = (float)(j - r), y = (float)(i - r);
      taper[(size_t)i * d + j] = 1.f - _smoothstep(sqrtf(x * x + y * y), 0.8f * (float)r, (float)r);
    }
  double total = 0.0;
  if(s->w_halo > 0.f)
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < d; i++)
      for(int j = 0; j < d; j++)
        part[(size_t)i * d + j] = _moffat((float)(j - r), (float)(i - r), s->halo_radius * diag_px,
                                          s->halo_beta, s->halo_stretch_x, s->halo_stretch_y)
                                  * taper[(size_t)i * d + j];
    if(_normalise(part, d)) { free(out); free(part); free(taper); return NULL; }
    _add_part(out, part, s->w_halo, n);
    total += s->w_halo;
  }
  if(s->w_veil > 0.f)
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < d; i++)
      for(int j = 0; j < d; j++)
        part[(size_t)i * d + j] = _moffat((float)(j - r), (float)(i - r), s->veil_radius * diag_px,
                                          s->veil_beta, 1.f, 1.f)
                                  * taper[(size_t)i * d + j];
    if(_normalise(part, d)) { free(out); free(part); free(taper); return NULL; }
    _add_part(out, part, s->w_veil, n);
    total += s->w_veil;
  }
  if(s->w_arms > 0.f && s->arms > 0)
  {
    if(_star(part, r, s, s->arm_radius * diag_px)) { free(out); free(part); free(taper); return NULL; }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t q = 0; q < n; q++) part[q] *= taper[q];
    if(_normalise(part, d)) { free(out); free(part); free(taper); return NULL; }
    _add_part(out, part, s->w_arms, n);
    total += s->w_arms;
  }
  free(part);
  free(taper);
  if(total <= 0.0)
  {
    memset(out, 0, sizeof(float) * n);
    out[(size_t)r * d + r] = 1.f;
    return out;
  }
  const float inv = (float)(1.0 / total);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(size_t q = 0; q < n; q++) out[q] *= inv;
  return out;
}

void ekvido_split_bands(const float *kernel, int r, double r_split_px, float *inner, float *outer)
{
  const int d = 2 * r + 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < d; i++)
    for(int j = 0; j < d; j++)
    {
      const size_t q = (size_t)i * d + j;
      const float x = (float)(j - r), y = (float)(i - r);
      const float rad = sqrtf(x * x + y * y);
      inner[q] = r_split_px <= 0.0 ? 0.f : kernel[q] * (1.f - ekvido_smoothstep(rad, 0.8 * r_split_px, 1.2 * r_split_px));
      outer[q] = kernel[q] - inner[q];
    }
}
