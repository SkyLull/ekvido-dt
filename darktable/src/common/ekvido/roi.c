#include "common/ekvido/roi.h"

static inline int _min(int a, int b) { return a < b ? a : b; }
static inline int _max(int a, int b) { return a > b ? a : b; }

ekvido_rect_t ekvido_expand_rect(ekvido_rect_t out, int margin, int full_w, int full_h)
{
  const int x0 = _min(out.x, _max(0, out.x - margin));
  const int y0 = _min(out.y, _max(0, out.y - margin));
  const int x1 = _max(out.x + out.w, _min(full_w, out.x + out.w + margin));
  const int y1 = _max(out.y + out.h, _min(full_h, out.y + out.h + margin));
  return (ekvido_rect_t){ x0, y0, x1 - x0, y1 - y0 };
}
