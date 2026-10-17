#pragma once
/* Linear (wrap-free) FFT convolution on a zero-padded grid. The grid API is the building block:
   spectra of planes and of centred kernels on one grid geometry, and the inverse back to the
   frame. ekvido_conv_* convolve fixed source planes with many kernels on top of it. Mirrors
   ekvido/common/convolve.py. */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* h×w frame, kernels up to half-size pad, on an fh×fw FFT grid (5-smooth, ≥ frame + pad). */
typedef struct ekvido_grid_t
{
  int h, w, pad, fh, fw, nthreads;
} ekvido_grid_t;

ekvido_grid_t ekvido_grid_make(int h, int w, int pad, int nthreads);
size_t ekvido_grid_spec_floats(const ekvido_grid_t *g); /* floats in one half-spectrum (interleaved complex) */
size_t ekvido_grid_floats(const ekvido_grid_t *g);      /* floats in the real grid; the `work` scratch size */
/* spectrum of an h×w plane, zero-padded */
void ekvido_grid_plane_spectrum(const ekvido_grid_t *g, const float *plane, float *work, float *spec);
/* spectrum of a (2r+1)² kernel centred at the origin; -1 (nothing written) if r > pad */
int ekvido_grid_kernel_spectrum(const ekvido_grid_t *g, const float *kernel, int r, float *work, float *spec);
/* inverse of spec (normalised), cropped to the h×w frame; spec is not modified */
void ekvido_grid_inverse(const ekvido_grid_t *g, const float *spec, float *work, float *out);

typedef struct ekvido_conv_t ekvido_conv_t;

ekvido_conv_t *ekvido_conv_new(const float *const *planes, int nplanes, int h, int w, int pad, int nthreads);
int ekvido_conv_apply(const ekvido_conv_t *c, const float *kernel, int r, float *const *out);
int ekvido_conv_pad(const ekvido_conv_t *c);
void ekvido_conv_free(ekvido_conv_t *c);
/* What a linear convolution of an all-ones h×w frame with the kernel gives (the share of the
   kernel landing inside the frame), from a summed-area table — no FFT. 0, -1 if r > pad, -2. */
int ekvido_conv_coverage(int h, int w, int pad, const float *kernel, int r, float *out, int nthreads);

#ifdef __cplusplus
}
#endif
