#include "common/ekvido/convolve.h"
#include "common/ekvido/fft.h"

#include <stdlib.h>
#include <string.h>

ekvido_grid_t ekvido_grid_make(int h, int w, int pad, int nthreads)
{
  ekvido_grid_t g = { h, w, pad, 0, 0, nthreads };
  g.fh = ekvido_fft_size(h + pad > 2 * pad + 1 ? h + pad : 2 * pad + 1);
  g.fw = ekvido_fft_size(w + pad > 2 * pad + 1 ? w + pad : 2 * pad + 1);
  return g;
}

size_t ekvido_grid_spec_floats(const ekvido_grid_t *g)
{
  return (size_t)g->fh * (size_t)(g->fw / 2 + 1) * 2;
}

size_t ekvido_grid_floats(const ekvido_grid_t *g)
{
  return (size_t)g->fh * (size_t)g->fw;
}

void ekvido_grid_plane_spectrum(const ekvido_grid_t *g, const float *plane, float *work, float *spec)
{
  memset(work, 0, sizeof(float) * ekvido_grid_floats(g));
  for(int i = 0; i < g->h; i++)
    memcpy(work + (size_t)i * g->fw, plane + (size_t)i * g->w, sizeof(float) * (size_t)g->w);
  ekvido_rfft2(work, spec, g->fh, g->fw, g->nthreads);
}

int ekvido_grid_kernel_spectrum(const ekvido_grid_t *g, const float *kernel, int r, float *work, float *spec)
{
  if(r > g->pad) return -1;
  memset(work, 0, sizeof(float) * ekvido_grid_floats(g));
  const int d = 2 * r + 1;
  /* kernel centred at the grid origin (Python: pad top-left, roll by -r) */
  for(int i = 0; i < d; i++)
    for(int j = 0; j < d; j++)
      work[(size_t)((i - r + g->fh) % g->fh) * g->fw + (size_t)((j - r + g->fw) % g->fw)] = kernel[(size_t)i * d + j];
  ekvido_rfft2(work, spec, g->fh, g->fw, g->nthreads);
  return 0;
}

void ekvido_grid_inverse(const ekvido_grid_t *g, const float *spec, float *work, float *out)
{
  ekvido_irfft2(spec, work, g->fh, g->fw, g->nthreads);
  for(int i = 0; i < g->h; i++)
    memcpy(out + (size_t)i * g->w, work + (size_t)i * g->fw, sizeof(float) * (size_t)g->w);
}

struct ekvido_conv_t
{
  ekvido_grid_t g;
  int n;
  float *spectra; /* n half-spectra */
};

void ekvido_conv_free(ekvido_conv_t *c)
{
  if(!c) return;
  free(c->spectra);
  free(c);
}

int ekvido_conv_pad(const ekvido_conv_t *c)
{
  return c->g.pad;
}

ekvido_conv_t *ekvido_conv_new(const float *const *planes, int n, int h, int w, int pad, int nthreads)
{
  ekvido_conv_t *c = calloc(1, sizeof(*c));
  if(!c) return NULL;
  c->g = ekvido_grid_make(h, w, pad, nthreads);
  c->n = n;
  const size_t sf = ekvido_grid_spec_floats(&c->g);
  c->spectra = malloc(sizeof(float) * sf * (size_t)n);
  float *work = malloc(sizeof(float) * ekvido_grid_floats(&c->g));
  if(!c->spectra || !work) { free(work); ekvido_conv_free(c); return NULL; }
  for(int p = 0; p < n; p++) ekvido_grid_plane_spectrum(&c->g, planes[p], work, c->spectra + sf * (size_t)p);
  free(work);
  return c;
}

int ekvido_conv_apply(const ekvido_conv_t *c, const float *kernel, int r, float *const *out)
{
  if(r > c->g.pad) return -1;
  const size_t sf = ekvido_grid_spec_floats(&c->g);
  float *work = malloc(sizeof(float) * ekvido_grid_floats(&c->g));
  float *kf = malloc(sizeof(float) * sf);
  float *prod = malloc(sizeof(float) * sf);
  if(!work || !kf || !prod) { free(work); free(kf); free(prod); return -2; }
  ekvido_grid_kernel_spectrum(&c->g, kernel, r, work, kf);
  for(int p = 0; p < c->n; p++)
  {
    const float *s = c->spectra + sf * (size_t)p;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t q = 0; q < sf; q += 2)
    {
      const float a = s[q], b = s[q + 1], e = kf[q], f = kf[q + 1];
      prod[q] = a * e - b * f;
      prod[q + 1] = a * f + b * e;
    }
    ekvido_grid_inverse(&c->g, prod, work, out[p]);
  }
  free(work); free(kf); free(prod);
  return 0;
}

/* out(y,x) = Σ kernel(i,j) over the offsets whose source pixel (y - (i-r), x - (j-r)) lies inside
   the frame: a rectangle of the kernel, summed from a double summed-area table. */
int ekvido_conv_coverage(int h, int w, int pad, const float *kernel, int r, float *out, int nthreads)
{
  (void)nthreads;
  if(r > pad) return -1;
  const int d = 2 * r + 1, sd = d + 1;
  double *sat = calloc((size_t)sd * sd, sizeof(double)); /* sat[i][j] = Σ kernel[<i][<j] */
  if(!sat) return -2;
  for(int i = 0; i < d; i++)
  {
    double row = 0.0;
    for(int j = 0; j < d; j++)
    {
      row += kernel[(size_t)i * d + j];
      sat[(size_t)(i + 1) * sd + (j + 1)] = sat[(size_t)i * sd + (j + 1)] + row;
    }
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int y = 0; y < h; y++)
  {
    const int i0 = y + r - h + 1 > 0 ? y + r - h + 1 : 0, i1 = y + r < d - 1 ? y + r : d - 1;
    for(int x = 0; x < w; x++)
    {
      const int j0 = x + r - w + 1 > 0 ? x + r - w + 1 : 0, j1 = x + r < d - 1 ? x + r : d - 1;
      const double s = sat[(size_t)(i1 + 1) * sd + (j1 + 1)] - sat[(size_t)i0 * sd + (j1 + 1)]
                       - sat[(size_t)(i1 + 1) * sd + j0] + sat[(size_t)i0 * sd + j0];
      out[(size_t)y * w + x] = (float)s;
    }
  }
  free(sat);
  return 0;
}
