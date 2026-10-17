#pragma once
#include "common/ekvido/psf.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void ekvido_kernel_log_u8(const float *k, size_t n, unsigned char *out);
float ekvido_kernel_view(const ekvido_kernel_spec_t *s, int width, unsigned char *bgra);
#ifdef __cplusplus
}
#endif
