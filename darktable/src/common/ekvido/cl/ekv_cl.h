#pragma once
/* ekvido GPU path (spec 2026-10-16 OpenCL §4.4): the Phase 1 CPU pipelines (kazuki.h, mist.h) on
   an OpenCL device, stage for stage. darktable-independent: darktable glue is ekv_cl_dt.c. */
#include "common/ekvido/cl/ekv_clfn.h"
#include "common/ekvido/kazuki.h"
#include "common/ekvido/mist.h"
#include "common/ekvido/roi.h"
#include <stddef.h>

typedef struct ekv_cl_t ekv_cl_t; /* one per device: program, kernels, FFT plans */

/* source: the text of data/kernels/ekvido.cl; plan_dir: where VkFFT plan binaries are kept (NULL =
   no disk cache). ctx/dev/q stay owned by the caller. NULL (with the reason on stderr) if the
   program does not build or a kernel is missing. */
ekv_cl_t *ekv_cl_new(cl_context ctx, cl_device_id dev, cl_command_queue q, const char *source, const char *plan_dir);
void ekv_cl_free(ekv_cl_t *E);
char *ekv_cl_read_source(const char *path); /* malloc'd, NUL-terminated file contents, or NULL */

/* ekvido_estimate_clip_level on an RGBA float image; NAN on a CL error */
float ekv_cl_estimate_clip_level(ekv_cl_t *E, cl_mem in, int w, int h);

typedef struct ekv_cl_cache_t ekv_cl_cache_t; /* per pipe piece: the cached glow (3 planes) on one device */
ekv_cl_cache_t *ekv_cl_cache_new(void);
void ekv_cl_cache_free(ekv_cl_cache_t *c); /* releases its device buffer */
int ekv_cl_cache_hits(const ekv_cl_cache_t *c);
/* in: RGBA float image2d iw×ih; out: RGBA float image2d rect.w×rect.h. Same contract as the CPU
   functions (kazuki.h, mist.h); 0, or -2 when a device allocation or OpenCL call failed (a
   failed run caches no new or partial glow, an earlier valid glow may stay, and `in` is never written). source_mask: host iw×ih floats or NULL. */
int ekv_cl_kazuki_process(ekv_cl_t *E, cl_mem in, int iw, int ih, cl_mem out, ekvido_rect_t rect,
                          const ekvido_kazuki_params_t *p, float diag_px, float clip_level,
                          const float *source_mask, int source_mask_invert, int pad_to_cap,
                          ekvido_kazuki_view_t view, ekv_cl_cache_t *cache);
int ekv_cl_mist_process(ekv_cl_t *E, cl_mem in, int iw, int ih, cl_mem out, ekvido_rect_t rect,
                        const ekvido_mist_params_t *p, float diag_px, int pad_to_cap, int glow_only,
                        ekv_cl_cache_t *cache);
/* out (RGBA image2d rect.w×rect.h) = the rect of in, exactly; 0 or -2 */
int ekv_cl_copy_rect(ekv_cl_t *E, cl_mem in, cl_mem out, ekvido_rect_t rect);
/* Peak device bytes of one run (input/output images and the cache excluded), for tiling_callback. */
size_t ekv_cl_kazuki_bytes(int iw, int ih, int pad);
size_t ekv_cl_mist_bytes(int iw, int ih, int pad);
