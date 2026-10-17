#include "common/ekvido/kernel_view.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void ekvido_kernel_log_u8(const float *k, size_t n, unsigned char *out)
{
  float peak = 0.f;
  for(size_t i = 0; i < n; i++) peak = fmaxf(peak, fmaxf(k[i], 0.f));
  peak = fmaxf(peak, 1e-30f);
  for(size_t i = 0; i < n; i++)
  {
    float v = (log10f(fmaxf(fmaxf(k[i], 0.f) / peak, 1e-6f)) + 6.f) / 6.f;
    v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    out[i] = (unsigned char)(v * 255.f + 0.5f);
  }
}

float ekvido_kernel_view(const ekvido_kernel_spec_t *s, int width, unsigned char *bgra)
{
  const size_t npx = (size_t)width * width;
  memset(bgra, 0, npx * 4);
  for(size_t i = 0; i < npx; i++) bgra[i * 4 + 3] = 255;
  double ext = fmin(ekvido_kernel_extent_frac(s), EKVIDO_MAX_KERNEL_FRAC);
  ext = fmax(ext, 1e-4);
  const int half = (width - 1) / 2;
  const float diag = (float)(half / ext); /* the whole kernel spans the pane */
  int r = 0;
  float *k = ekvido_build_kernel(s, diag, &r);
  if(!k) return diag;
  const int d = 2 * r + 1;
  unsigned char *g = malloc((size_t)d * d);
  if(g)
  {
    ekvido_kernel_log_u8(k, (size_t)d * d, g);
    for(int i = 0; i < d; i++)
      for(int j = 0; j < d; j++)
      {
        const int X = half + j - r, Y = half + i - r;
        if(X < 0 || Y < 0 || X >= width || Y >= width) continue;
        unsigned char *px = bgra + ((size_t)Y * width + X) * 4;
        px[0] = px[1] = px[2] = g[(size_t)i * d + j];
      }
    free(g);
  }
  free(k);
  return diag;
}
