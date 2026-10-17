#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ekvido_rect_t { int x, y, w, h; } ekvido_rect_t;
/* Grow `out` by `margin` on every side, clamped to the full image; never smaller than `out`. */
ekvido_rect_t ekvido_expand_rect(ekvido_rect_t out, int margin, int full_w, int full_h);
#ifdef __cplusplus
}
#endif
