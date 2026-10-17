#pragma once
/* Kazuki ("Kazuki light") core, mirroring ekvido/modules/kazuki.py: highlight energy,
   per-source warmth, a 7-plane glow basis whose spectra are multiplied by the inner/outer
   kernel-band spectra and recoloured by the warm/cool ramps in the frequency domain (3 inverse
   FFTs), then laid on by the blend mode. Same maths as the Python 9-plane formulation. */
#include "common/ekvido/blend.h"
#include "common/ekvido/psf.h"
#include "common/ekvido/roi.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* GUI-only preview (spec §5.4); exports always use FINAL. */
typedef enum ekvido_kazuki_view_t
{
  EKVIDO_VIEW_FINAL = 0,
  EKVIDO_VIEW_GLOW = 1,
  EKVIDO_VIEW_ENERGY = 2,
  EKVIDO_VIEW_WARMTH = 3,
  EKVIDO_VIEW_CLIPPED = 4,
} ekvido_kazuki_view_t;

typedef struct ekvido_kazuki_params_t
{
  float strength;
  ekvido_blend_mode_t blend;
  float blend_white_ev;
  float threshold_ev, knee_ev;
  float hue_radius, neutral_point, warmth_softness, neutral_bias;
  float push_warm, push_cool;
  float warm_core[3], warm_edge[3], cool_core[3], cool_edge[3]; /* linear Rec.2020 RGB */
  float band_split;
  ekvido_kernel_spec_t spec;
} ekvido_kazuki_params_t;

#define EKVIDO_KAZUKI_NB 7
/* Basis plane k of [e_r, e_g, e_b, t·e_r, t·e_g, t·e_b, t²·Y(e)] into out (np floats). Y(e) and
   t·Y(e), the Python basis' other two planes, are sums of these and live in the coefficients. */
void ekvido_kazuki_basis_plane(const float *const e[3], const float *t, size_t np, int k, float *out);
/* glow_c = Σ_k Σ_band coef[band][c][k] · (basis_k ⊛ band kernel); band 0 inner (core colours),
   1 outer (edge colours). The warm/cool recolouring of Python's _combine, folded. */
void ekvido_kazuki_coefficients(const ekvido_kazuki_params_t *p, float coef[2][3][EKVIDO_KAZUKI_NB]);
/* Input margin a zoomed view needs: kernel half-size + the warmth blur's reach, capped. */
int ekvido_kazuki_margin(const ekvido_kazuki_params_t *p, float diag_px);

typedef struct ekvido_kazuki_cache_t ekvido_kazuki_cache_t;
ekvido_kazuki_cache_t *ekvido_kazuki_cache_new(void);
void ekvido_kazuki_cache_free(ekvido_kazuki_cache_t *c);
/* How often the source stage, the kernel-band spectra and the finished glow were reused. */
void ekvido_kazuki_cache_hits(const ekvido_kazuki_cache_t *c, int *source_hits, int *kernel_hits, int *glow_hits);

/* in: iw×ih RGBA; out: rect.w×rect.h RGBA, rect inside in. clip_level: resolved by the
   caller (auto estimate or 2^clip_ev; INFINITY = nothing clipped). pad_to_cap: pad the
   FFT grid to the 0.25-diagonal cap so cached spectra survive shape changes.
   Returns 0, or -2 when out of memory. */
int ekvido_kazuki_process(const float *in, int iw, int ih, float *out, ekvido_rect_t rect,
                          const ekvido_kazuki_params_t *p, float diag_px, float clip_level, int pad_to_cap,
                          ekvido_kazuki_view_t view, ekvido_kazuki_cache_t *cache, int nthreads);

/* The same with a source mask (spec §5.7): source_mask is iw×ih (the module's whole
   input, margin included), values 0…1; NULL = no mask, identical to ekvido_kazuki_process.
   Only the energy is masked (E' = M·E): warmth and clip level still come from the whole
   input; conserve subtracts E'; the energy view shows E'. */
int ekvido_kazuki_process_masked(const float *in, int iw, int ih, float *out, ekvido_rect_t rect,
                                 const ekvido_kazuki_params_t *p, float diag_px, float clip_level,
                                 const float *source_mask, int source_mask_invert, int pad_to_cap,
                                 ekvido_kazuki_view_t view, ekvido_kazuki_cache_t *cache, int nthreads);

#ifdef __cplusplus
}
#endif
