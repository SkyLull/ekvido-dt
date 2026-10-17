#include "common/ekvido/wblur.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EKVIDO_DIRECT_MAX_SIGMA 8.0

static inline int _clampi(int v, int lo, int hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

/* Separable blur, horizontal then vertical, replicate padding (Python F.pad "replicate"). */
static int _direct(const float *in, float *out, int h, int w, double sigma)
{
  const int r = (int)fmax(1.0, ceil(3.0 * sigma));
  const int n = 2 * r + 1;
  float *g = malloc(sizeof(float) * (size_t)n);
  float *tmp = malloc(sizeof(float) * (size_t)h * w);
  if(!g || !tmp) { free(g); free(tmp); return -2; }
  const float sf = (float)sigma;
  float gs = 0.f;
  for(int t = 0; t < n; t++)
  {
    const float x = (float)(t - r) / sf;
    g[t] = expf(-0.5f * x * x);
    gs += g[t];
  }
  for(int t = 0; t < n; t++) g[t] /= gs;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < h; i++)
    for(int j = 0; j < w; j++)
    {
      float acc = 0.f;
      for(int t = 0; t < n; t++) acc += g[t] * in[(size_t)i * w + _clampi(j + t - r, 0, w - 1)];
      tmp[(size_t)i * w + j] = acc;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < h; i++)
    for(int j = 0; j < w; j++)
    {
      float acc = 0.f;
      for(int t = 0; t < n; t++) acc += g[t] * tmp[(size_t)_clampi(i + t - r, 0, h - 1) * w + j];
      out[(size_t)i * w + j] = acc;
    }
  free(g);
  free(tmp);
  return 0;
}

/* Box-downsample by f (replicate-padded to a multiple of f), blur small, upsample back. */
static int _downsampled(const float *in, float *out, int h, int w, double sigma)
{
  const int f = (int)floor(sigma / 4.0);
  const int ph = (f - h % f) % f, pw = (f - w % f) % f;
  const int H = h + ph, W = w + pw, sh = H / f, sw = W / f;
  float *small = malloc(sizeof(float) * (size_t)sh * sw);
  if(!small) return -2;
  const float inv = 1.f / (float)(f * f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int y = 0; y < sh; y++)
    for(int x = 0; x < sw; x++)
    {
      float acc = 0.f;
      for(int dy = 0; dy < f; dy++)
        for(int dx = 0; dx < f; dx++)
          acc += in[(size_t)_clampi(y * f + dy, 0, h - 1) * w + _clampi(x * f + dx, 0, w - 1)];
      small[(size_t)y * sw + x] = acc * inv;
    }
  if(ekvido_gaussian_blur(small, small, sh, sw, sigma / f)) { free(small); return -2; }
  /* PyTorch bilinear, align_corners = False: src = (dst + 0.5) * in/out - 0.5, clamped at 0 */
  const float scale_y = (float)sh / (float)H, scale_x = (float)sw / (float)W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i = 0; i < h; i++)
  {
    const float sy = fmaxf(scale_y * ((float)i + 0.5f) - 0.5f, 0.f);
    const int y0 = (int)sy, y1 = y0 + (y0 < sh - 1 ? 1 : 0);
    const float ly1 = sy - (float)y0, ly0 = 1.f - ly1;
    for(int j = 0; j < w; j++)
    {
      const float sx = fmaxf(scale_x * ((float)j + 0.5f) - 0.5f, 0.f);
      const int x0 = (int)sx, x1 = x0 + (x0 < sw - 1 ? 1 : 0);
      const float lx1 = sx - (float)x0, lx0 = 1.f - lx1;
      const float *r0 = small + (size_t)y0 * sw, *r1 = small + (size_t)y1 * sw;
      out[(size_t)i * w + j] = ly0 * (lx0 * r0[x0] + lx1 * r0[x1]) + ly1 * (lx0 * r1[x0] + lx1 * r1[x1]);
    }
  }
  free(small);
  return 0;
}

int ekvido_gaussian_blur(const float *in, float *out, int h, int w, double sigma_px)
{
  if(sigma_px <= 0.0)
  {
    if(out != in) memmove(out, in, sizeof(float) * (size_t)h * w);
    return 0;
  }
  if(sigma_px > EKVIDO_DIRECT_MAX_SIGMA) return _downsampled(in, out, h, w, sigma_px);
  return _direct(in, out, h, w, sigma_px);
}

int ekvido_weighted_blur(const float *const *x, int nx, const float *wt, int h, int w, double sigma_px,
                         float *const *out, float *den)
{
  const size_t np = (size_t)h * w;
  if(ekvido_gaussian_blur(wt, den, h, w, sigma_px)) return -2;
  for(int k = 0; k < nx; k++)
  {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t i = 0; i < np; i++) out[k][i] = x[k][i] * wt[i];
    if(ekvido_gaussian_blur(out[k], out[k], h, w, sigma_px)) return -2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t i = 0; i < np; i++) out[k][i] /= fmaxf(den[i], 1e-12f);
  }
  return 0;
}
