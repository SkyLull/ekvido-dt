#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_OPENCL
#include "common/ekvido/cl/ekv_cl_priv.h"
#include "common/ekvido/cl/ekv_clredirect.h"
#include "common/ekvido/fft.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const ekv_clfn_t *ekv_clfn = NULL;
int ekv_cl_fail_alloc_at = 0;

void ekv_cl_set_api(const ekv_clfn_t *fn)
{
  ekv_clfn = fn;
}

char *ekv_cl_read_source(const char *path)
{
  FILE *f = path ? fopen(path, "rb") : NULL;
  if(!f) return NULL;
  char *s = NULL;
  if(!fseek(f, 0, SEEK_END))
  {
    const long n = ftell(f);
    if(n > 0 && !fseek(f, 0, SEEK_SET) && (s = malloc((size_t)n + 1)))
    {
      if(fread(s, 1, (size_t)n, f) == (size_t)n)
        s[n] = '\0';
      else
      {
        free(s);
        s = NULL;
      }
    }
  }
  fclose(f);
  return s;
}

void ekv_cl_free(ekv_cl_t *E)
{
  if(!E) return;
  ekv_cl_fft_end(E);
  ekv_cl_fft_free_plans(E);
  for(int i = 0; i < EKV_K_COUNT; i++)
    if(E->k[i]) clReleaseKernel(E->k[i]);
  if(E->prog) clReleaseProgram(E->prog);
  ekv_cl_release(E->dummy);
  free(E->plan_dir);
  free(E);
}

ekv_cl_t *ekv_cl_new(cl_context ctx, cl_device_id dev, cl_command_queue q, const char *source, const char *plan_dir)
{
  if(!ekv_clfn || !source) return NULL;
  ekv_cl_t *E = calloc(1, sizeof(*E));
  if(!E) return NULL;
  E->ctx = ctx;
  E->dev = dev;
  E->q = q;
  cl_int err = clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(E->platform), &E->platform, NULL);
  char name[96] = { 0 }, driver[64] = { 0 };
  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
  clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(driver) - 1, driver, NULL);
  snprintf(E->devkey, sizeof(E->devkey), "%s-%s", name, driver);
  for(char *c = E->devkey; *c; c++)
    if(!isalnum((unsigned char)*c) && *c != '.' && *c != '-') *c = '_';
  E->plan_dir = plan_dir ? strdup(plan_dir) : NULL;
  if(err == CL_SUCCESS) E->prog = clCreateProgramWithSource(ctx, 1, &source, NULL, &err);
  if(!E->prog || clBuildProgram(E->prog, 1, &dev, "-cl-std=CL1.2", NULL, NULL) != CL_SUCCESS)
  {
    char log[8192] = { 0 };
    if(E->prog) clGetProgramBuildInfo(E->prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
    fprintf(stderr, "[ekvido] ekvido.cl does not build (%d):\n%s\n", (int)err, log);
    ekv_cl_free(E);
    return NULL;
  }
  static const char *names[EKV_K_COUNT] = {
#define EKV_K_NAME(n) "ekv_" #n,
    EKV_KERNELS(EKV_K_NAME)
#undef EKV_K_NAME
  };
  for(int i = 0; i < EKV_K_COUNT; i++)
  {
    E->k[i] = clCreateKernel(E->prog, names[i], &err);
    if(err != CL_SUCCESS)
    {
      fprintf(stderr, "[ekvido] kernel %s missing (%d)\n", names[i], (int)err);
      E->k[i] = NULL;
      ekv_cl_free(E);
      return NULL;
    }
  }
  E->dummy = ekv_cl_buffer(E, 16);
  if(!E->dummy)
  {
    ekv_cl_free(E);
    return NULL;
  }
  return E;
}

cl_mem ekv_cl_buffer(ekv_cl_t *E, const size_t bytes)
{
  if(ekv_cl_fail_alloc_at > 0 && --ekv_cl_fail_alloc_at == 0) return NULL;
  cl_int err = CL_SUCCESS;
  cl_mem m = clCreateBuffer(E->ctx, CL_MEM_READ_WRITE, bytes ? bytes : 4, NULL, &err);
  return err == CL_SUCCESS ? m : NULL;
}

void ekv_cl_release(cl_mem m)
{
  if(m) clReleaseMemObject(m);
}

int ekv_cl_upload(ekv_cl_t *E, cl_mem b, const void *host, const size_t bytes)
{
  return clEnqueueWriteBuffer(E->q, b, CL_TRUE, 0, bytes, host, 0, NULL, NULL) == CL_SUCCESS ? 0 : -2;
}

int ekv_cl_download(ekv_cl_t *E, cl_mem b, void *host, const size_t bytes)
{
  return clEnqueueReadBuffer(E->q, b, CL_TRUE, 0, bytes, host, 0, NULL, NULL) == CL_SUCCESS ? 0 : -2;
}

int ekv_cl_run(ekv_cl_t *E, const int kernel, const size_t gx, const size_t gy, const int nargs, ...)
{
  if(gx == 0) return 0;
  cl_kernel k = E->k[kernel];
  cl_int err = CL_SUCCESS;
  va_list ap;
  va_start(ap, nargs);
  for(int a = 0; a < nargs && err == CL_SUCCESS; a++)
  {
    const size_t size = va_arg(ap, size_t);
    const void *ptr = va_arg(ap, const void *);
    err = clSetKernelArg(k, (cl_uint)a, size, ptr);
  }
  va_end(ap);
  if(err != CL_SUCCESS) return -2;
  const size_t g[2] = { gx, gy ? gy : 1 };
  return clEnqueueNDRangeKernel(E->q, k, gy ? 2 : 1, NULL, g, NULL, 0, NULL, NULL) == CL_SUCCESS ? 0 : -2;
}

static int _grid_dim(const int n, const int pad)
{
  const int m = n + pad > 2 * pad + 1 ? n + pad : 2 * pad + 1;
  return ekvido_fft_size(m > 512 ? (m + 255) / 256 * 256 : m);
}

ekv_cl_grid_t ekv_cl_grid_make(const int h, const int w, const int pad)
{
  ekv_cl_grid_t g = { h, w, pad, _grid_dim(h, pad), _grid_dim(w, pad), 0 };
  g.stride = 2 * (g.fw / 2 + 1);
  return g;
}

size_t ekv_cl_grid_bytes(const ekv_cl_grid_t *g)
{
  return sizeof(float) * (size_t)g->fh * (size_t)g->stride;
}

size_t ekv_cl_grid_bins(const ekv_cl_grid_t *g)
{
  return (size_t)g->fh * (size_t)(g->fw / 2 + 1);
}

#define EKV_DIRECT_MAX_SIGMA 8.0 /* = EKVIDO_DIRECT_MAX_SIGMA in wblur.c */

typedef struct ekv_ss_t
{
  float lo, span;
  int flat;
} ekv_ss_t;

/* ekvido_smoothstep's float bounds, computed as the CPU does (double lo/hi, cast at use) */
static ekv_ss_t _ss(const double lo, const double hi)
{
  const ekv_ss_t s = { (float)lo, (float)(hi - lo), hi <= lo };
  return s;
}

int ekv_cl_split(ekv_cl_t *E, cl_mem in, const int w, const int h, cl_mem src)
{
  return ekv_cl_run(E, EKV_K_split, (size_t)w, (size_t)h, 4, EKV_ARG(in), EKV_ARG(w), EKV_ARG(h), EKV_ARG(src));
}

int ekv_cl_energy(ekv_cl_t *E, cl_mem src, const size_t np, const float thr, const float knee, cl_mem energy)
{
  const ekv_ss_t s = _ss((double)thr - knee, (double)thr + knee);
  const cl_uint n = (cl_uint)np;
  return ekv_cl_run(E, EKV_K_energy, np, 0, 6, EKV_ARG(src), EKV_ARG(energy), EKV_ARG(n), EKV_ARG(s.lo),
                    EKV_ARG(s.span), EKV_ARG(s.flat));
}

int ekv_cl_mask_energy(ekv_cl_t *E, cl_mem energy, cl_mem mask, const size_t np, const int invert)
{
  const cl_uint n = (cl_uint)np;
  const int inv = invert != 0;
  return ekv_cl_run(E, EKV_K_mask_energy, np, 0, 4, EKV_ARG(energy), EKV_ARG(mask), EKV_ARG(n), EKV_ARG(inv));
}

float ekv_cl_estimate_clip_level(ekv_cl_t *E, cl_mem in, const int w, const int h)
{
  if(w <= 0 || h <= 0) return INFINITY;
  const size_t np = (size_t)w * h;
  float *top = malloc(sizeof(float) * (size_t)h);
  cl_uint *cnt = malloc(sizeof(cl_uint) * (size_t)h);
  cl_mem pm = ekv_cl_buffer(E, sizeof(float) * (size_t)h), pc = ekv_cl_buffer(E, sizeof(cl_uint) * (size_t)h);
  float res = NAN;
  if(top && cnt && pm && pc
     && !ekv_cl_run(E, EKV_K_row_max, (size_t)h, 0, 4, EKV_ARG(in), EKV_ARG(w), EKV_ARG(h), EKV_ARG(pm))
     && !ekv_cl_download(E, pm, top, sizeof(float) * (size_t)h))
  {
    float t = 0.f;
    for(int y = 0; y < h; y++) t = fmaxf(t, top[y]);
    if(!(t > 0.f))
      res = INFINITY;
    else
    {
      const float near = (float)((double)t * (1.0 - 1e-5));
      if(!ekv_cl_run(E, EKV_K_row_count, (size_t)h, 0, 5, EKV_ARG(in), EKV_ARG(w), EKV_ARG(h), EKV_ARG(near), EKV_ARG(pc))
         && !ekv_cl_download(E, pc, cnt, sizeof(cl_uint) * (size_t)h))
      {
        size_t count = 0;
        for(int y = 0; y < h; y++) count += cnt[y];
        res = (double)count / (double)np > 1e-4 ? (float)(0.999 * (double)t) : INFINITY;
      }
    }
  }
  ekv_cl_release(pm);
  ekv_cl_release(pc);
  free(top);
  free(cnt);
  return res;
}

uint64_t ekv_cl_hash_image(ekv_cl_t *E, cl_mem in, const int w, const int h, int *err)
{
  cl_ulong *rows = malloc(sizeof(cl_ulong) * (size_t)h);
  cl_mem part = ekv_cl_buffer(E, sizeof(cl_ulong) * (size_t)h);
  *err = !rows || !part || ekv_cl_run(E, EKV_K_row_hash, (size_t)h, 0, 4, EKV_ARG(in), EKV_ARG(w), EKV_ARG(h), EKV_ARG(part))
         || ekv_cl_download(E, part, rows, sizeof(cl_ulong) * (size_t)h);
  uint64_t hh = 1469598103934665603ULL ^ ((uint64_t)(uint32_t)w << 32 | (uint32_t)h);
  if(!*err)
    for(int y = 0; y < h; y++) { hh ^= rows[y]; hh *= 1099511628211ULL; }
  ekv_cl_release(part);
  free(rows);
  return hh;
}

static int _blur_direct(ekv_cl_t *E, cl_mem in, cl_mem out, const int h, const int w, const double sigma)
{
  const int r = (int)fmax(1.0, ceil(3.0 * sigma)), n = 2 * r + 1;
  float g[64]; /* sigma <= 8 here: n <= 49 */
  const float sf = (float)sigma;
  float gs = 0.f;
  for(int t = 0; t < n; t++)
  {
    const float x = (float)(t - r) / sf;
    g[t] = expf(-0.5f * x * x);
    gs += g[t];
  }
  for(int t = 0; t < n; t++) g[t] /= gs;
  cl_mem taps = ekv_cl_buffer(E, sizeof(float) * n), tmp = ekv_cl_buffer(E, sizeof(float) * (size_t)h * w);
  const int err = !taps || !tmp || ekv_cl_upload(E, taps, g, sizeof(float) * n)
                  || ekv_cl_run(E, EKV_K_blur_h, (size_t)w, (size_t)h, 6, EKV_ARG(in), EKV_ARG(tmp), EKV_ARG(h), EKV_ARG(w),
                                EKV_ARG(taps), EKV_ARG(r))
                  || ekv_cl_run(E, EKV_K_blur_v, (size_t)w, (size_t)h, 6, EKV_ARG(tmp), EKV_ARG(out), EKV_ARG(h), EKV_ARG(w),
                                EKV_ARG(taps), EKV_ARG(r));
  ekv_cl_release(taps); /* released after the queued kernels finish (OpenCL semantics) */
  ekv_cl_release(tmp);
  return err ? -2 : 0;
}

static int _blur_down(ekv_cl_t *E, cl_mem in, cl_mem out, const int h, const int w, const double sigma)
{
  const int f = (int)floor(sigma / 4.0);
  const int ph = (f - h % f) % f, pw = (f - w % f) % f;
  const int H = h + ph, W = w + pw, sh = H / f, sw = W / f;
  const float inv = 1.f / (float)(f * f);
  const float scale_y = (float)sh / (float)H, scale_x = (float)sw / (float)W;
  cl_mem small = ekv_cl_buffer(E, sizeof(float) * (size_t)sh * sw);
  const int err = !small
                  || ekv_cl_run(E, EKV_K_box_down, (size_t)sw, (size_t)sh, 8, EKV_ARG(in), EKV_ARG(small), EKV_ARG(h),
                                EKV_ARG(w), EKV_ARG(f), EKV_ARG(sh), EKV_ARG(sw), EKV_ARG(inv))
                  || ekv_cl_gaussian_blur(E, small, small, sh, sw, sigma / f)
                  || ekv_cl_run(E, EKV_K_bilinear_up, (size_t)w, (size_t)h, 8, EKV_ARG(small), EKV_ARG(out), EKV_ARG(h),
                                EKV_ARG(w), EKV_ARG(sh), EKV_ARG(sw), EKV_ARG(scale_y), EKV_ARG(scale_x));
  ekv_cl_release(small);
  return err ? -2 : 0;
}

int ekv_cl_gaussian_blur(ekv_cl_t *E, cl_mem in, cl_mem out, const int h, const int w, const double sigma)
{
  if(sigma <= 0.0)
  {
    const cl_uint n = (cl_uint)((size_t)h * w);
    return in == out ? 0 : ekv_cl_run(E, EKV_K_copy, n, 0, 3, EKV_ARG(in), EKV_ARG(out), EKV_ARG(n));
  }
  return sigma > EKV_DIRECT_MAX_SIGMA ? _blur_down(E, in, out, h, w, sigma) : _blur_direct(E, in, out, h, w, sigma);
}

int ekv_cl_weighted_blur(ekv_cl_t *E, const cl_mem *x, const int nx, cl_mem wt, const int h, const int w,
                         const double sigma, const cl_mem *out, cl_mem den)
{
  const size_t np = (size_t)h * w;
  const cl_uint n = (cl_uint)np;
  if(ekv_cl_gaussian_blur(E, wt, den, h, w, sigma)) return -2;
  for(int k = 0; k < nx; k++)
    if(ekv_cl_run(E, EKV_K_mul, np, 0, 4, EKV_ARG(x[k]), EKV_ARG(wt), EKV_ARG(out[k]), EKV_ARG(n))
       || ekv_cl_gaussian_blur(E, out[k], out[k], h, w, sigma)
       || ekv_cl_run(E, EKV_K_div_floor, np, 0, 3, EKV_ARG(out[k]), EKV_ARG(den), EKV_ARG(n)))
      return -2;
  return 0;
}

int ekv_cl_warmth_map(ekv_cl_t *E, cl_mem src, const int h, const int w, const float clip_level, const double sigma_px,
                      const float y_floor, const float neutral_point, const float softness, const float bias, cl_mem t,
                      cl_mem clipped)
{
  const size_t np = (size_t)h * w;
  const cl_uint n = (cl_uint)np;
  cl_mem b[7];
  int err = 0;
  for(int i = 0; i < 7; i++) err |= !(b[i] = ekv_cl_buffer(E, sizeof(float) * np));
  cl_mem wt = b[0], xr = b[1], xb = b[2], cm = b[3], hr = b[4], hb = b[5], den = b[6];
  const cl_mem x[2] = { xr, xb }, hue[2] = { hr, hb };
  const ekv_ss_t ws = _ss(-(double)softness, (double)softness);
  err = err
        || ekv_cl_run(E, EKV_K_warmth_prep, np, 0, 8, EKV_ARG(src), EKV_ARG(n), EKV_ARG(clip_level), EKV_ARG(wt),
                      EKV_ARG(xr), EKV_ARG(xb), EKV_ARG(cm), EKV_ARG(clipped))
        || ekv_cl_weighted_blur(E, x, 2, wt, h, w, sigma_px, hue, den)
        || ekv_cl_gaussian_blur(E, cm, cm, h, w, sigma_px)
        || ekv_cl_run(E, EKV_K_warmth_finish, np, 0, 12, EKV_ARG(den), EKV_ARG(cm), EKV_ARG(hr), EKV_ARG(hb), EKV_ARG(n),
                      EKV_ARG(y_floor), EKV_ARG(neutral_point), EKV_ARG(ws.lo), EKV_ARG(ws.span), EKV_ARG(ws.flat),
                      EKV_ARG(bias), EKV_ARG(t));
  for(int i = 0; i < 7; i++) ekv_cl_release(b[i]);
  return err ? -2 : 0;
}

int ekv_cl_grid_basis(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem e, cl_mem t, const int k, cl_mem grid)
{
  return ekv_cl_run(E, EKV_K_basis_grid, (size_t)g->stride, (size_t)g->fh, 8, EKV_ARG(e), EKV_ARG(t), EKV_ARG(g->w),
                    EKV_ARG(g->h), EKV_ARG(k), EKV_ARG(grid), EKV_ARG(g->fh), EKV_ARG(g->stride));
}

int ekv_cl_grid_kernel(ekv_cl_t *E, const ekv_cl_grid_t *g, const float *kernel, const int r, cl_mem grid)
{
  if(r > g->pad) return -1;
  const int d = 2 * r + 1;
  const size_t kb = sizeof(float) * (size_t)d * d;
  const cl_uint n = (cl_uint)(ekv_cl_grid_bytes(g) / sizeof(float));
  cl_mem kd = ekv_cl_buffer(E, kb);
  const int err = !kd || ekv_cl_upload(E, kd, kernel, kb) || ekv_cl_run(E, EKV_K_zero, n, 0, 2, EKV_ARG(grid), EKV_ARG(n))
                  || ekv_cl_run(E, EKV_K_place, (size_t)d, (size_t)d, 6, EKV_ARG(kd), EKV_ARG(r), EKV_ARG(grid),
                                EKV_ARG(g->fh), EKV_ARG(g->fw), EKV_ARG(g->stride))
                  || ekv_cl_fft(E, g, grid, 0);
  ekv_cl_release(kd);
  return err ? -2 : 0;
}

int ekv_cl_mac(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem s, cl_mem ki, cl_mem ko,
               float coef[2][3][EKVIDO_KAZUKI_NB], const int k, const cl_mem acc[3])
{
  const cl_uint nc = (cl_uint)ekv_cl_grid_bins(g);
  return ekv_cl_run(E, EKV_K_mac, nc, 0, 14, EKV_ARG(s), EKV_ARG(ki), EKV_ARG(ko), EKV_ARG(nc), EKV_ARG(k),
                    EKV_ARG(coef[0][0][k]), EKV_ARG(coef[0][1][k]), EKV_ARG(coef[0][2][k]), EKV_ARG(coef[1][0][k]),
                    EKV_ARG(coef[1][1][k]), EKV_ARG(coef[1][2][k]), EKV_ARG(acc[0]), EKV_ARG(acc[1]), EKV_ARG(acc[2]));
}

int ekv_cl_cmul(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem s, cl_mem kf)
{
  const cl_uint nc = (cl_uint)ekv_cl_grid_bins(g);
  return ekv_cl_run(E, EKV_K_cmul, nc, 0, 3, EKV_ARG(s), EKV_ARG(kf), EKV_ARG(nc));
}

int ekv_cl_crop(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem grid, cl_mem out, const size_t off)
{
  const float scale = (float)(1.0 / ((double)g->fh * (double)g->fw));
  const cl_uint o = (cl_uint)off;
  return ekv_cl_run(E, EKV_K_crop, (size_t)g->w, (size_t)g->h, 7, EKV_ARG(grid), EKV_ARG(g->stride), EKV_ARG(g->w),
                    EKV_ARG(g->h), EKV_ARG(scale), EKV_ARG(out), EKV_ARG(o));
}

int ekv_cl_div_cover(ekv_cl_t *E, cl_mem glow, cl_mem cover, const size_t np)
{
  const cl_uint n = (cl_uint)np;
  return ekv_cl_run(E, EKV_K_div_cover, np, 0, 3, EKV_ARG(glow), EKV_ARG(cover), EKV_ARG(n));
}

int ekv_cl_copy_rect(ekv_cl_t *E, cl_mem in, cl_mem out, const ekvido_rect_t rect)
{
  const size_t src[3] = { (size_t)rect.x, (size_t)rect.y, 0 }, dst[3] = { 0, 0, 0 };
  const size_t reg[3] = { (size_t)rect.w, (size_t)rect.h, 1 };
  return clEnqueueCopyImage(E->q, in, out, src, dst, reg, 0, NULL, NULL) == CL_SUCCESS ? 0 : -2;
}

int ekv_cl_kazuki_out(ekv_cl_t *E, cl_mem in, cl_mem out, const ekvido_rect_t rect, const int iw, const int ih,
                      const ekvido_kazuki_view_t view, cl_mem glow, cl_mem energy, cl_mem t, cl_mem clipped,
                      const ekvido_blend_mode_t mode, const float strength, const float white)
{
  (void)ih;
  const cl_uint np = (cl_uint)((size_t)iw * ih);
  const int v = (int)view, m = (int)mode;
  cl_mem gl = glow ? glow : E->dummy, en = energy ? energy : E->dummy;
  cl_mem tt = t ? t : E->dummy, cm = clipped ? clipped : E->dummy;
  return ekv_cl_run(E, EKV_K_kazuki_out, (size_t)rect.w, (size_t)rect.h, 16, EKV_ARG(in), EKV_ARG(out), EKV_ARG(rect.x),
                    EKV_ARG(rect.y), EKV_ARG(rect.w), EKV_ARG(rect.h), EKV_ARG(iw), EKV_ARG(np), EKV_ARG(v), EKV_ARG(gl),
                    EKV_ARG(en), EKV_ARG(tt), EKV_ARG(cm), EKV_ARG(m), EKV_ARG(strength), EKV_ARG(white));
}

int ekv_cl_mist_out(ekv_cl_t *E, cl_mem in, cl_mem out, const ekvido_rect_t rect, const int iw, const int ih,
                    cl_mem glow, const float k, const int glow_only)
{
  (void)ih;
  const cl_uint np = (cl_uint)((size_t)iw * ih);
  const int go = glow_only != 0;
  return ekv_cl_run(E, EKV_K_mist_out, (size_t)rect.w, (size_t)rect.h, 11, EKV_ARG(in), EKV_ARG(out), EKV_ARG(rect.x),
                    EKV_ARG(rect.y), EKV_ARG(rect.w), EKV_ARG(rect.h), EKV_ARG(iw), EKV_ARG(np), EKV_ARG(glow),
                    EKV_ARG(k), EKV_ARG(go));
}

#endif /* HAVE_OPENCL */
