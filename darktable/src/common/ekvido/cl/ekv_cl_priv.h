#pragma once
/* Internals of the ekvido GPU path, shared by its .c files and the GPU tests. */
#include "common/ekvido/cl/ekv_cl.h"
#include "common/ekvido/blend.h"
#include <stdint.h>

/* Kernels of ekvido.cl, each built as "ekv_<name>". */
#define EKV_KERNELS(X)                                                                              \
  X(zero) X(copy) X(split) X(energy) X(mask_energy) X(row_max) X(row_count) X(row_hash) X(warmth_prep) \
  X(warmth_finish) X(blur_h) X(blur_v) X(box_down) X(bilinear_up) X(mul) X(div_floor) \
  X(basis_grid) X(place) X(mac) X(cmul) X(crop) X(div_cover) X(kazuki_out) X(mist_out)

enum
{
#define EKV_K_ENUM(n) EKV_K_##n,
  EKV_KERNELS(EKV_K_ENUM)
#undef EKV_K_ENUM
  EKV_K_COUNT
};

#define EKV_PLANS 2             /* VkFFT plans kept per device (least recently used goes) */
#define EKV_CL_FFT_TEMP_GRIDS 1 /* VkFFT scratch of the FFT phase, in grids (spec §10.7; checked by the GPU tests) */

typedef struct ekv_cl_plan_t
{
  int fh, fw;
  unsigned tick;
  uint64_t bytes; /* the grid buffer size the plan was made for */
  cl_mem buf;     /* buffer handed to initializeVkFFT; launches pass their own */
  void *app;      /* VkFFTApplication * */
  double ms;      /* creation time */
  int from_disk;  /* loaded from the plan cache */
  uint64_t temp;  /* scratch its launches need in E->fft_temp (bytes), 0 if none */
} ekv_cl_plan_t;

struct ekv_cl_t
{
  cl_context ctx;
  cl_device_id dev;
  cl_command_queue q;
  cl_platform_id platform;
  cl_program prog;
  cl_kernel k[EKV_K_COUNT];
  cl_mem dummy; /* 16 bytes, bound to buffer arguments a launch does not read */
  ekv_cl_plan_t plan[EKV_PLANS];
  unsigned tick;
  cl_mem fft_temp;       /* VkFFT scratch, alive from the first FFT that needs it to ekv_cl_fft_end */
  uint64_t fft_temp_bytes;
  char *plan_dir;
  char devkey[200]; /* device name + driver version, file-name safe */
};

/* In-place R2C grid for an h×w frame padded by pad: fh, fw as ekvido_grid_make (max(n+pad,
   2·pad+1), 5-smooth) but rounded up to a multiple of 256 above 512 (spec §4.4); rows of
   stride = 2·(fw/2+1) floats, so the forward FFT leaves the CPU's half-spectrum layout. */
typedef struct ekv_cl_grid_t
{
  int h, w, pad, fh, fw, stride;
} ekv_cl_grid_t;
ekv_cl_grid_t ekv_cl_grid_make(int h, int w, int pad);
size_t ekv_cl_grid_bytes(const ekv_cl_grid_t *g); /* fh·stride floats */
size_t ekv_cl_grid_bins(const ekv_cl_grid_t *g);  /* fh·(fw/2+1) complex bins */

cl_mem ekv_cl_buffer(ekv_cl_t *E, size_t bytes);  /* NULL on failure */
void ekv_cl_release(cl_mem m);                    /* NULL-safe */
int ekv_cl_upload(ekv_cl_t *E, cl_mem b, const void *host, size_t bytes);  /* blocking; 0 or -2 */
int ekv_cl_download(ekv_cl_t *E, cl_mem b, void *host, size_t bytes);      /* blocking; 0 or -2 */
/* Launch kernel EKV_K_x over gx (1-D, gy == 0) or gx×gy; nargs pairs (size_t size, const void *ptr). */
int ekv_cl_run(ekv_cl_t *E, int kernel, size_t gx, size_t gy, int nargs, ...);
#define EKV_ARG(x) sizeof(x), (const void *)&(x)

/* ekv_cl_fft.c: in place on a grid buffer; forward leaves the half-spectrum, inverse the
   unnormalised real grid. 0 or -2. */
int ekv_cl_fft(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem buf, int inverse);
void ekv_cl_fft_end(ekv_cl_t *E); /* releases the FFT scratch; plans (kernels only) stay */
void ekv_cl_fft_free_plans(ekv_cl_t *E);
void ekv_cl_fft_plan_path(const ekv_cl_t *E, int fh, int fw, char *path, size_t n); /* plan file of fh×fw */

/* Source stage (ekv_cl.c); every function returns 0 or -2 unless noted */
int ekv_cl_split(ekv_cl_t *E, cl_mem in, int w, int h, cl_mem src3);
int ekv_cl_energy(ekv_cl_t *E, cl_mem src3, size_t np, float thr, float knee, cl_mem energy3);
int ekv_cl_mask_energy(ekv_cl_t *E, cl_mem energy3, cl_mem mask, size_t np, int invert);
uint64_t ekv_cl_hash_image(ekv_cl_t *E, cl_mem in, int w, int h, int *err);
int ekv_cl_gaussian_blur(ekv_cl_t *E, cl_mem in, cl_mem out, int h, int w, double sigma); /* in may equal out */
int ekv_cl_weighted_blur(ekv_cl_t *E, const cl_mem *x, int nx, cl_mem wt, int h, int w, double sigma, const cl_mem *out,
                         cl_mem den);
int ekv_cl_warmth_map(ekv_cl_t *E, cl_mem src3, int h, int w, float clip_level, double sigma_px, float y_floor,
                      float neutral_point, float softness, float bias, cl_mem t, cl_mem clipped); /* clipped: uchar/pixel */

/* Grid and output stages (ekv_cl.c); 0 or -2 unless noted */
/* basis plane k (with e3 = source planes and k < 3, plane k) into the zero-padded grid */
int ekv_cl_grid_basis(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem e3, cl_mem t, int k, cl_mem grid);
/* host kernel placed centred/wrapped, forward FFT done; -1 if r > pad */
int ekv_cl_grid_kernel(ekv_cl_t *E, const ekv_cl_grid_t *g, const float *kernel, int r, cl_mem grid);
int ekv_cl_mac(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem s, cl_mem ki, cl_mem ko, float coef[2][3][EKVIDO_KAZUKI_NB],
               int k, const cl_mem acc[3]);
int ekv_cl_cmul(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem s, cl_mem kf); /* s *= kf (complex, in place) */
/* out[off + y·w + x] = grid[y·stride + x] / (fh·fw) */
int ekv_cl_crop(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem grid, cl_mem out, size_t off);
int ekv_cl_div_cover(ekv_cl_t *E, cl_mem glow3, cl_mem cover, size_t np);
/* NULL buffers allowed for what the view does not read */
int ekv_cl_kazuki_out(ekv_cl_t *E, cl_mem in, cl_mem out, ekvido_rect_t rect, int iw, int ih, ekvido_kazuki_view_t view,
                      cl_mem glow, cl_mem energy, cl_mem t, cl_mem clipped, ekvido_blend_mode_t mode, float strength,
                      float white);
int ekv_cl_mist_out(ekv_cl_t *E, cl_mem in, cl_mem out, ekvido_rect_t rect, int iw, int ih, cl_mem glow, float k,
                    int glow_only);

/* Tests only: the Nth ekv_cl_buffer call from now fails (0 = never). Not thread-safe; never set in darktable. */
extern int ekv_cl_fail_alloc_at;
