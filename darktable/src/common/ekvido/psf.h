#pragma once
/* Glow kernel (PSF): halo + veil Moffats and an L_p star, mirroring ekvido/common/psf.py.
   Sizes are fractions of the image diagonal. */
#ifdef __cplusplus
extern "C" {
#endif

#define EKVIDO_MAX_KERNEL_FRAC 0.25

typedef struct ekvido_kernel_spec_t
{
  float halo_radius, halo_beta, halo_stretch_x, halo_stretch_y;
  float veil_radius, veil_beta;
  int arms;
  float arm_angle, arm_radius, arm_beta, arm_sharpness, arm_stretch_x, arm_stretch_y;
  float w_halo, w_veil, w_arms;
} ekvido_kernel_spec_t;

int ekvido_kernel_half_cap(float diag_px);
double ekvido_kernel_extent_frac(const ekvido_kernel_spec_t *s);
int ekvido_kernel_half_size(const ekvido_kernel_spec_t *s, float diag_px);
/* malloc'd (2r+1)² kernel, sum 1, r in *radius; NULL when out of memory. */
float *ekvido_build_kernel(const ekvido_kernel_spec_t *s, float diag_px, int *radius);

/* Inner/outer radial bands with a soft ±20 % edge at r_split_px; inner + outer == kernel.
   kernel, inner, outer are (2r+1)^2. Mirrors psf.split_bands. */
void ekvido_split_bands(const float *kernel, int r, double r_split_px, float *inner, float *outer);

#ifdef __cplusplus
}
#endif
