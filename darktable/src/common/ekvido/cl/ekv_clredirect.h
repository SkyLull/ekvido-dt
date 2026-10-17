#pragma once
/* After ekv_clfn.h (and after <CL/cl.h> has been included): route every cl* call in this
   translation unit through ekvido's table. */
#define clBuildProgram ekv_clfn->BuildProgram
#define clCreateBuffer ekv_clfn->CreateBuffer
#define clCreateCommandQueue ekv_clfn->CreateCommandQueue
#define clCreateKernel ekv_clfn->CreateKernel
#define clCreateProgramWithBinary ekv_clfn->CreateProgramWithBinary
#define clCreateProgramWithSource ekv_clfn->CreateProgramWithSource
#define clEnqueueCopyImage ekv_clfn->EnqueueCopyImage
#define clEnqueueNDRangeKernel ekv_clfn->EnqueueNDRangeKernel
#define clEnqueueReadBuffer ekv_clfn->EnqueueReadBuffer
#define clEnqueueWriteBuffer ekv_clfn->EnqueueWriteBuffer
#define clFinish ekv_clfn->Finish
#define clGetDeviceInfo ekv_clfn->GetDeviceInfo
#define clGetProgramBuildInfo ekv_clfn->GetProgramBuildInfo
#define clGetProgramInfo ekv_clfn->GetProgramInfo
#define clReleaseCommandQueue ekv_clfn->ReleaseCommandQueue
#define clReleaseKernel ekv_clfn->ReleaseKernel
#define clReleaseMemObject ekv_clfn->ReleaseMemObject
#define clReleaseProgram ekv_clfn->ReleaseProgram
#define clSetKernelArg ekv_clfn->SetKernelArg
