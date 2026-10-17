#pragma once
/* OpenCL entry points of the ekvido GPU path, called through a table: darktable loads OpenCL at
   run time (common/dlopencl.c) and never links it, so neither this code nor the vendored VkFFT
   may call the real symbols. darktable fills the table from its loaded symbols (ekv_cl_dt.c);
   the standalone GPU tests from the linked library (tests/cl/clfn_direct.c). Files that call
   cl* by name include ekv_clredirect.h after this header. */
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#include <CL/cl.h>

/* The 18 calls VkFFT 1.3.4 makes, plus clEnqueueCopyImage (all in darktable's dlopencl table). */
#define EKV_CLFN_LIST(X)                                                                          \
  X(BuildProgram) X(CreateBuffer) X(CreateCommandQueue) X(CreateKernel) X(CreateProgramWithBinary)  \
  X(CreateProgramWithSource) X(EnqueueCopyImage) X(EnqueueNDRangeKernel) X(EnqueueReadBuffer)       \
  X(EnqueueWriteBuffer) X(Finish) X(GetDeviceInfo) X(GetProgramBuildInfo) X(GetProgramInfo)         \
  X(ReleaseCommandQueue) X(ReleaseKernel) X(ReleaseMemObject) X(ReleaseProgram) X(SetKernelArg)

/* clCreateCommandQueue is deprecated from OpenCL 2.0; darktable includes cl.h at a newer target
   version before this header, so gcc flags the typeof below (-Werror). The table needs it. */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
typedef struct ekv_clfn_t
{
#define EKV_CLFN_MEMBER(n) __typeof__(&cl##n) n;
  EKV_CLFN_LIST(EKV_CLFN_MEMBER)
#undef EKV_CLFN_MEMBER
} ekv_clfn_t;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

extern const ekv_clfn_t *ekv_clfn;
/* Set once, before any other ekv_cl call; fn must outlive every ekv_cl_t. */
void ekv_cl_set_api(const ekv_clfn_t *fn);
