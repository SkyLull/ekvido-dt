#pragma once
/* Kazuki glow blend modes, mirroring ekvido/common/blend.py (spec §5.6). The scalar
   functions work on one channel; OVERLAY_GLOW needs the whole pixel (its layer opacity is
   the largest channel of the glow), so pixels go through ekvido_blend_px. */
#ifdef __cplusplus
extern "C" {
#endif

/* Same order as Python BLEND_MODES and the module's params enum; append only. */
typedef enum ekvido_blend_mode_t
{
  EKVIDO_BLEND_ADDITIVE = 0,
  EKVIDO_BLEND_CONSERVE = 1,
  EKVIDO_BLEND_SCREEN = 2,
  EKVIDO_BLEND_OVERLAY = 3,
  EKVIDO_BLEND_COLOR_DODGE = 4,
  EKVIDO_BLEND_GLOW_DODGE = 5,
  EKVIDO_BLEND_OVERLAY_GLOW = 6,
} ekvido_blend_mode_t;

/* overlay_glow: a layer opacity m at or below this counts as no glow (Python GLOW_EPS) */
#define EKVIDO_GLOW_EPS 1e-6f

float ekvido_srgb_encode(float x); /* clamps to 0…1 first (Python srgb_oetf) */
float ekvido_srgb_decode(float v); /* clamps to 0…1 first (Python srgb_eotf) */

/* CSP colour dodge D(A, C): 0 if A = 0, else 1 if C ≥ 1, else min(1, A / (1 − C)). */
float ekvido_dodge(float a, float c);
/* Spec §5.6 table on display-encoded values for screen, overlay and the dodges
   (OVERLAY_GLOW: one channel with m = c; use ekvido_overlay_glow for a real pixel). */
float ekvido_csp_blend(ekvido_blend_mode_t mode, float a, float c, float alpha);
/* overlay (glow only), one channel: A + α·m·(O(A, C/m) − A); exactly A where m ≤ EKVIDO_GLOW_EPS. */
float ekvido_overlay_glow(float a, float c, float m, float alpha);

/* Kazuki output for one channel: img with the glow (before strength) laid on by mode.
   white = 2^blend_white_ev, used by the CSP modes only. */
float ekvido_blend(ekvido_blend_mode_t mode, float img, float glow, float energy, float strength, float white);
/* overlay (glow only) is a pixel-only mode: called here with EKVIDO_BLEND_OVERLAY_GLOW this returns img
   unchanged. Use ekvido_blend_px. */
/* The same for one RGB pixel — the per-pixel stage the module uses (any mode). */
void ekvido_blend_px(ekvido_blend_mode_t mode, const float img[3], const float glow[3], const float energy[3],
                     float strength, float white, float out[3]);

#ifdef __cplusplus
}
#endif
