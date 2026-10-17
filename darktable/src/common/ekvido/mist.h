#pragma once
/* Mist (diffusion filter) core, mirroring ekvido/modules/mist.py. */
#include "common/ekvido/psf.h"
#include "common/ekvido/roi.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ekvido_mist_params_t
{
  float strength;
  ekvido_kernel_spec_t spec;
} ekvido_mist_params_t;

typedef struct ekvido_mist_cache_t ekvido_mist_cache_t;

ekvido_mist_cache_t *ekvido_mist_cache_new(void);
void ekvido_mist_cache_free(ekvido_mist_cache_t *c);
int ekvido_mist_cache_glow_hits(const ekvido_mist_cache_t *c);

int ekvido_mist_process(const float *in, int iw, int ih, float *out, ekvido_rect_t rect,
                        const ekvido_mist_params_t *p, float diag_px, int pad_to_cap, int glow_only,
                        ekvido_mist_cache_t *cache, int nthreads);

#ifdef __cplusplus
}
#endif
