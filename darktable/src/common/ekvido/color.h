#pragma once
/* Kazuki colour stages, mirroring ekvido/common/color.py and kazuki._warmth_map. */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* HSV with V = 1 (Python colorsys, in double), hue in degrees. */
void ekvido_hsv_to_rgb(float hue_deg, float sat, float rgb[3]);
/* rgb / its Rec.2020 luminance (floor 1e-8). */
void ekvido_unit_luminance(const float rgb[3], float out[3]);

/* Warmth map t (0 cool … 1 warm) and the clipped mask from positive RGB planes.
   sigma_px = hue_radius × diagonal; y_floor = 2^threshold_ev. Returns 0 or -2. */
int ekvido_warmth_map(const float *const src[3], int h, int w, float clip_level, double sigma_px,
                      float y_floor, float neutral_point, float softness, float bias, float *t,
                      unsigned char *clipped);

/* Preview colour for warmth t (browser colormap: blue / grey / orange, 8-bit sRGB),
   decoded to linear so the rest of the pipe shows roughly those colours. */
void ekvido_warmth_colour(float t, float rgb[3]);

#ifdef __cplusplus
}
#endif
