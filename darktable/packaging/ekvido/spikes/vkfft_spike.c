/* S1 spike: VkFFT (OpenCL backend) R2C + C2R on an fh×fw grid vs PocketFFT (fft.cc).
   usage: vkfft_spike FH FW   — prints plan time, forward/inverse ms (median of 5) and errors. */
#define VKFFT_BACKEND 3
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "vkFFT.h"

#include "common/ekvido/fft.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp(const void *a, const void *b)
{
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
  if(argc != 3) { fprintf(stderr, "usage: %s FH FW\n", argv[0]); return 2; }
  const uint64_t fh = (uint64_t)atoll(argv[1]), fw = (uint64_t)atoll(argv[2]), wc = fw / 2 + 1;
  const size_t real_n = (size_t)(fh * fw), spec_n = (size_t)(fh * wc * 2);
  float *x = malloc(sizeof(float) * real_n), *ref_spec = malloc(sizeof(float) * spec_n);
  float *padded = malloc(sizeof(float) * spec_n), *back = malloc(sizeof(float) * spec_n);
  if(!x || !ref_spec || !padded || !back) { fprintf(stderr, "host out of memory\n"); return 1; }
  uint32_t s = 7u;
  for(size_t i = 0; i < real_n; i++) { s = s * 1664525u + 1013904223u; x[i] = (float)(s >> 8) / 16777216.f; }
  const double t_ref = now();
  ekvido_rfft2(x, ref_spec, (int)fh, (int)fw, 16);
  const double t_ref_ms = (now() - t_ref) * 1e3;
  /* in-place R2C layout: rows padded to 2*wc floats */
  memset(padded, 0, sizeof(float) * spec_n);
  for(uint64_t i = 0; i < fh; i++) memcpy(padded + i * 2 * wc, x + i * fw, sizeof(float) * fw);

  cl_platform_id platform; cl_device_id device; cl_int e;
  if(clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS) { fprintf(stderr, "no OpenCL platform\n"); return 1; }
  if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS) { fprintf(stderr, "no GPU\n"); return 1; }
  char name[256] = { 0 };
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
  cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &e);
  cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &e);
  uint64_t bytes = sizeof(float) * spec_n;
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &e);
  if(e != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer %d (%.0f MB)\n", e, bytes / 1048576.0); return 1; }

  VkFFTConfiguration cfg = { 0 };
  cfg.FFTdim = 2;
  cfg.size[0] = fw; /* fastest axis first */
  cfg.size[1] = fh;
  cfg.performR2C = 1;
  cfg.platform = &platform;
  cfg.device = &device;
  cfg.context = &ctx;
  cfg.bufferSize = &bytes;
  cfg.buffer = &buf;
  VkFFTApplication app = { 0 };
  const double t_plan = now();
  VkFFTResult r = initializeVkFFT(&app, cfg);
  const double plan_ms = (now() - t_plan) * 1e3;
  if(r != VKFFT_SUCCESS) { fprintf(stderr, "initializeVkFFT failed: %d\n", (int)r); return 1; }
  VkFFTLaunchParams lp = { 0 };
  lp.commandQueue = &q;
  lp.buffer = &buf;

  double fwd[5], inv[5];
  for(int it = 0; it < 6; it++) /* first round warms up, not counted */
  {
    clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, padded, 0, NULL, NULL);
    clFinish(q);
    double t0 = now();
    r = VkFFTAppend(&app, -1, &lp);
    clFinish(q);
    double t1 = now();
    if(it == 1) clEnqueueReadBuffer(q, buf, CL_TRUE, 0, bytes, back, 0, NULL, NULL); /* forward result */
    r |= VkFFTAppend(&app, 1, &lp);
    clFinish(q);
    double t2 = now();
    if(r != VKFFT_SUCCESS) { fprintf(stderr, "VkFFTAppend failed: %d\n", (int)r); return 1; }
    if(it) { fwd[it - 1] = (t1 - t0) * 1e3; inv[it - 1] = (t2 - t1) * 1e3; }
  }
  /* forward vs PocketFFT, relative to the spectrum peak */
  double peak = 0.0, ferr = 0.0;
  for(size_t i = 0; i < spec_n; i++) peak = fmax(peak, fabs(ref_spec[i]));
  for(size_t i = 0; i < spec_n; i++) ferr = fmax(ferr, fabs((double)back[i] - ref_spec[i]));
  /* round trip (VkFFT inverse is unnormalised) vs the input */
  clEnqueueReadBuffer(q, buf, CL_TRUE, 0, bytes, back, 0, NULL, NULL);
  double rerr = 0.0;
  const double norm = 1.0 / ((double)fh * (double)fw);
  for(uint64_t i = 0; i < fh; i++)
    for(uint64_t j = 0; j < fw; j++) rerr = fmax(rerr, fabs(back[i * 2 * wc + j] * norm - x[i * fw + j]));
  qsort(fwd, 5, sizeof(double), cmp);
  qsort(inv, 5, sizeof(double), cmp);
  printf("%s %llux%llu: plan %.0f ms, forward %.1f ms, inverse %.1f ms (PocketFFT forward %.0f ms, 16 thr); "
         "forward max err %.2e of peak, round-trip max err %.2e, buffer %.0f MB\n",
         name, (unsigned long long)fh, (unsigned long long)fw, plan_ms, fwd[2], inv[2], t_ref_ms, ferr / peak, rerr,
         bytes / 1048576.0);
  deleteVkFFT(&app);
  clReleaseMemObject(buf);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);
  free(x); free(ref_spec); free(padded); free(back);
  return 0;
}
