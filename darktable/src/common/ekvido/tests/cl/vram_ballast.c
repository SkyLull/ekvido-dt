/* Holds GB gigabytes of GPU memory for SECONDS (CPU-fallback test: stands in for another
   process holding GPU memory). usage: vram_ballast GB SECONDS */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
  if(argc != 3) { fprintf(stderr, "usage: %s GB SECONDS\n", argv[0]); return 2; }
  const int gb = atoi(argv[1]), secs = atoi(argv[2]);
  if(gb <= 0 || secs <= 0) { fprintf(stderr, "usage: %s GB SECONDS (both > 0)\n", argv[0]); return 2; }
  cl_platform_id pl[8]; cl_uint np = 0; cl_device_id dev = NULL;
  clGetPlatformIDs(8, pl, &np);
  for(cl_uint i = 0; i < np && !dev; i++) clGetDeviceIDs(pl[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
  if(!dev) { fprintf(stderr, "no GPU\n"); return 1; }
  cl_int e;
  cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
  if(!ctx || e != CL_SUCCESS) { fprintf(stderr, "clCreateContext failed (%d)\n", e); return 1; }
  cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);
  if(!q || e != CL_SUCCESS) { fprintf(stderr, "clCreateCommandQueue failed (%d)\n", e); return 1; }
  const size_t chunk = (size_t)512 << 20;
  char *zeros = calloc(1, chunk);
  int held = 0;
  for(int i = 0; i < gb * 2; i++)
  {
    cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, chunk, NULL, &e);
    if(e != CL_SUCCESS || clEnqueueWriteBuffer(q, m, CL_TRUE, 0, chunk, zeros, 0, NULL, NULL) != CL_SUCCESS) break;
    held++; /* buffers are kept until exit on purpose */
  }
  printf("holding %.1f GB\n", held / 2.0);
  fflush(stdout);
  sleep((unsigned)secs);
  free(zeros);
  if(held < gb * 2)
  {
    fprintf(stderr, "could only hold %.1f of %d GB\n", held / 2.0, gb);
    return 1;
  }
  return 0;
}
