/* Before any header: VkFFT writes every constant of its generated kernels with
   sprintf("%.17Le", (long double)x) (vkFFT_MathUtils.h:671, 906, ...). MinGW's default printf
   family is the Windows CRT's, whose long double is 64-bit, so it misreads gcc's 80-bit long
   double and the kernels get garbage twiddles (~1e-312: every FFT returns ~0). darktable's build
   defines this for all Windows sources (src/CMakeLists.txt); the standalone GPU tests do not. */
#if defined(__MINGW32__) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 1
#endif
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_OPENCL
/* The only VkFFT translation unit (compiled with -w: VkFFT is not warning-clean). Plans: one per
   (fh, fw), EKV_PLANS kept per device; their compiled binaries are cached on disk under
   E->plan_dir, keyed by VkFFT version + configuration tag + device + driver + size (spec §8 S1:
   0.5–1.3 s per plan). Plans hold kernels only: VkFFT's scratch (userTempBuffer) is E->fft_temp,
   allocated by the first FFT that needs it and released by ekv_cl_fft_end at the end of the
   pipeline's FFT phase, so an idle darktable holds no FFT scratch. */
#include "common/ekvido/cl/ekv_cl_priv.h"
#include "common/ekvido/cl/ekv_clredirect.h"
/* VkFFT includes <OpenCL/opencl.h> on Apple, which clashes with the <CL/cl.h> (darktable's own
   headers) that ekv_clfn.h already pulled in: pre-empt Apple's include guard. */
#ifdef __APPLE__
#define __OPENCL_H
#endif
#define VKFFT_BACKEND 3
#include "external/vkfft/vkFFT.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <process.h>
#define EKV_GETPID _getpid
#else
#include <unistd.h>
#define EKV_GETPID getpid
#endif

#define EKV_VKFFT_VERSION "vkfft134"
/* What _init asks VkFFT for (2-D in-place R2C, user temp buffer): bump on any change to _init's
   configuration so plan files built for another configuration are never loaded. */
#define EKV_PLAN_TAG "r2c-ut1"

static double _now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 1e3 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
}

static void _plan_free(ekv_cl_plan_t *p)
{
  if(p->app)
  {
    deleteVkFFT((VkFFTApplication *)p->app);
    free(p->app);
  }
  memset(p, 0, sizeof(*p));
}

void ekv_cl_fft_free_plans(ekv_cl_t *E)
{
  for(int i = 0; i < EKV_PLANS; i++) _plan_free(&E->plan[i]);
}

void ekv_cl_fft_end(ekv_cl_t *E)
{
  ekv_cl_release(E->fft_temp); /* enqueued FFTs keep it alive until they finish (OpenCL refcount) */
  E->fft_temp = NULL;
  E->fft_temp_bytes = 0;
}

void ekv_cl_fft_plan_path(const ekv_cl_t *E, const int fh, const int fw, char *path, const size_t n)
{
  snprintf(path, n, "%s/%s-%s-%s-%dx%d.bin", E->plan_dir ? E->plan_dir : ".", EKV_VKFFT_VERSION, EKV_PLAN_TAG,
           E->devkey, fh, fw);
}

/* A plan file is handed to VkFFT only if it is at least VkFFT's 5-word header and its first word
   (the total size VkFFT wrote, vkFFT_InitializeApp.h:1776) equals the file size: a truncated or
   foreign file is deleted and the plan rebuilt. */
static int _blob_ok(const void *blob, const size_t size)
{
  uint64_t total = 0;
  if(!blob || size < 5 * sizeof(uint64_t)) return 0;
  memcpy(&total, blob, sizeof(total));
  return total == (uint64_t)size;
}

static void *_read_file(const char *path, size_t *size)
{
  FILE *f = fopen(path, "rb");
  if(!f) return NULL;
  void *b = NULL;
  if(!fseek(f, 0, SEEK_END))
  {
    const long n = ftell(f);
    if(n > 0 && !fseek(f, 0, SEEK_SET) && (b = malloc((size_t)n)) && fread(b, 1, (size_t)n, f) != (size_t)n)
    {
      free(b);
      b = NULL;
    }
    if(b) *size = (size_t)n;
  }
  fclose(f);
  return b;
}

static void _write_file(const char *path, const void *data, const size_t size)
{
  char tmp[1100];
  snprintf(tmp, sizeof(tmp), "%s.%ld.tmp", path, (long)EKV_GETPID()); /* two darktables never share a tmp */
  FILE *f = fopen(tmp, "wb");
  if(!f) return;
  const int ok = fwrite(data, 1, size, f) == size;
  if(fclose(f) || !ok)
  {
    remove(tmp);
    return;
  }
#ifdef _WIN32
  remove(path); /* Windows rename does not replace an existing file */
#endif
  if(rename(tmp, path)) remove(tmp);
}

/* Scratch this plan's launches touch: VkFFT binds the temp buffer only to the uploads of an axis
   that takes more than one upload (vkFFT_UpdateBuffers.h:852, the condition that also makes VkFFT
   allocate one itself, vkFFT_Plan_FFT.h:227), and sizes it per axis as the complex size of that
   axis's view of the data (vkFFT_Scheduler.h:2902-2916, single precision, one batch): axis 0 of
   an even R2C too long for shared memory is split in half (bigSequenceEvenR2C,
   vkFFT_Scheduler.h:2259-2268), else fw complex; axis 1 sees fw/2+1 columns
   (vkFFT_Scheduler.h:2281-2282). VkFFT's own allocation is the max over all axes, including
   single-upload ones that never read it (2 grids for a portrait grid whose short axis 0 fits in
   shared memory); this is the max over the axes that do. 0: no axis reads it. */
static uint64_t _temp_bytes(const VkFFTApplication *app, const int fh, const int fw)
{
  const uint64_t c = 2 * sizeof(float), axis1 = (uint64_t)(fw / 2 + 1) * (uint64_t)fh * c;
  const VkFFTPlan *plans[2] = { app->localFFTPlan, app->localFFTPlan_inverse };
  uint64_t need = 0;
  for(int k = 0; k < 2; k++)
  {
    const VkFFTPlan *pl = plans[k];
    if(!pl) continue;
    const uint64_t axis0 = (uint64_t)(pl->bigSequenceEvenR2C ? fw / 2 + 1 : fw) * (uint64_t)fh * c;
    if(pl->numAxisUploads[0] > 1 && axis0 > need) need = axis0;
    if(pl->numAxisUploads[1] > 1 && axis1 > need) need = axis1;
  }
  return need;
}

static VkFFTResult _init(ekv_cl_t *E, ekv_cl_plan_t *p, VkFFTApplication *app, void *blob)
{
  VkFFTConfiguration cfg = { 0 };
  cfg.FFTdim = 2;
  cfg.size[0] = (pfUINT)p->fw; /* fastest axis first */
  cfg.size[1] = (pfUINT)p->fh;
  cfg.performR2C = 1;
  cfg.platform = &E->platform;
  cfg.device = &E->dev;
  cfg.context = &E->ctx;
  cfg.bufferSize = &p->bytes;
  cfg.buffer = &p->buf;
  /* Scratch comes with each launch (VkFFTLaunchParams.tempBuffer), never allocated by VkFFT. Its
     declared size must be at least what VkFFT computes or init fails
     (VKFFT_ERROR_INVALID_user_tempBuffer_too_small, vkFFT_Scheduler.h:2940-2942): 8·fw·fh bytes
     bounds every axis of a non-Bluestein grid (ours are 5-smooth). With one temp buffer this size
     only feeds VkFFT's multi-buffer split (vkFFT_UpdateBuffers.h:880-886), so the buffer each
     launch passes may be the smaller _temp_bytes. VkFFT owns and frees this array
     (vkFFT_DeleteApp.h:115-117); a temp buffer at init is optional (vkFFT_UpdateBuffers.h:688). */
  cfg.userTempBuffer = 1;
  cfg.tempBufferSize = malloc(sizeof(pfUINT));
  if(!cfg.tempBufferSize) return VKFFT_ERROR_MALLOC_FAILED;
  cfg.tempBufferSize[0] = (pfUINT)8 * (pfUINT)p->fw * (pfUINT)p->fh;
  if(blob)
  {
    cfg.loadApplicationFromString = 1;
    cfg.loadApplicationString = blob;
  }
  else if(E->plan_dir)
    cfg.saveApplicationToString = 1;
  return initializeVkFFT(app, cfg);
}

/* VkFFT app for an fh×fw in-place R2C grid; buf is a grid buffer of that size (any). */
static VkFFTApplication *_plan(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem buf)
{
  ekv_cl_plan_t *slot = NULL, *old = &E->plan[0];
  for(int i = 0; i < EKV_PLANS; i++)
  {
    ekv_cl_plan_t *p = &E->plan[i];
    if(p->app && p->fh == g->fh && p->fw == g->fw)
    {
      p->tick = ++E->tick;
      return p->app;
    }
    if(!p->app && !slot) slot = p;
    if(p->tick < old->tick) old = p;
  }
  if(!slot)
  {
    _plan_free(old);
    slot = old;
  }
  VkFFTApplication *app = calloc(1, sizeof(*app));
  if(!app) return NULL;
  slot->fh = g->fh;
  slot->fw = g->fw;
  slot->bytes = (uint64_t)ekv_cl_grid_bytes(g);
  slot->buf = buf;
  const double t0 = _now_ms();
  char path[1024] = { 0 };
  size_t blob_size = 0;
  void *blob = NULL;
  if(E->plan_dir)
  {
    ekv_cl_fft_plan_path(E, g->fh, g->fw, path, sizeof(path));
    blob = _read_file(path, &blob_size);
    if(blob && !_blob_ok(blob, blob_size)) /* truncated or foreign: never handed to VkFFT */
    {
      free(blob);
      blob = NULL;
      remove(path);
    }
  }
  VkFFTResult r = _init(E, slot, app, blob);
  if(r != VKFFT_SUCCESS && blob) /* plan VkFFT itself rejects: rebuild */
  {
    deleteVkFFT(app);
    memset(app, 0, sizeof(*app));
    r = _init(E, slot, app, NULL);
    free(blob);
    blob = NULL;
  }
  if(r != VKFFT_SUCCESS)
  {
    free(blob);
    deleteVkFFT(app);
    free(app);
    memset(slot, 0, sizeof(*slot));
    return NULL;
  }
  slot->from_disk = blob != NULL;
  free(blob);
  if(!slot->from_disk && E->plan_dir && app->saveApplicationString && app->applicationStringSize)
    _write_file(path, app->saveApplicationString, (size_t)app->applicationStringSize);
  slot->ms = _now_ms() - t0;
  slot->temp = _temp_bytes(app, g->fh, g->fw);
  slot->app = app;
  slot->tick = ++E->tick;
  return app;
}

int ekv_cl_fft(ekv_cl_t *E, const ekv_cl_grid_t *g, cl_mem buf, const int inverse)
{
  VkFFTApplication *app = _plan(E, g, buf);
  if(!app) return -2;
  const ekv_cl_plan_t *p = NULL;
  for(int i = 0; i < EKV_PLANS && !p; i++)
    if(E->plan[i].app == app) p = &E->plan[i];
  if(!p) return -2;
  if(p->temp > E->fft_temp_bytes)
  {
    ekv_cl_fft_end(E);
    if(!(E->fft_temp = ekv_cl_buffer(E, (size_t)p->temp))) return -2;
    E->fft_temp_bytes = p->temp;
  }
  VkFFTLaunchParams lp = { 0 };
  lp.commandQueue = &E->q;
  lp.buffer = &buf; /* every launch names its buffer: the plan's own may have been released */
  /* VkFFT dereferences these cl_mem pointers when it sets kernel arguments (vkFFT_DispatchPlan.h:
     298-306), so the current E->fft_temp is used even when the pointer is unchanged; a plan that
     never reads its temp buffer gets the 16-byte dummy (a launch must name one,
     vkFFT_UpdateBuffers.h:705-706). */
  lp.tempBuffer = p->temp ? &E->fft_temp : &E->dummy;
  return VkFFTAppend(app, inverse ? 1 : -1, &lp) == VKFFT_SUCCESS ? 0 : -2;
}
#endif /* HAVE_OPENCL */
