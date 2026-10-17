#pragma once
/* Gaussian and weighted blur, mirroring ekvido/common/wblur.py exactly: separable direct
   blur with replicate edges up to sigma 8 px; beyond that box-downsample by
   f = floor(sigma / 4), blur small, and bilinear-upsample (align_corners = False). */
#ifdef __cplusplus
extern "C" {
#endif

/* One H*W plane; `out` may equal `in`. Returns 0, or -2 when out of memory. */
int ekvido_gaussian_blur(const float *in, float *out, int h, int w, double sigma_px);

/* out[k] = blur(x[k] * wt) / max(blur(wt), 1e-12) for k < nx; den = blur(wt). */
int ekvido_weighted_blur(const float *const *x, int nx, const float *wt, int h, int w, double sigma_px,
                         float *const *out, float *den);

#ifdef __cplusplus
}
#endif
