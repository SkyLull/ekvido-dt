#pragma once
/* darktable side of the ekvido GPU path: OpenCL table from darktable's loaded symbols, one
   ekv_cl_t per device (built on first use), released when the last ekvido module unloads. */
#ifdef HAVE_OPENCL
#include "common/ekvido/cl/ekv_cl.h"

ekv_cl_t *ekv_cl_dt_get(int devid); /* NULL: use the CPU path on this device */
void ekv_cl_dt_retain(void);        /* module init_global */
void ekv_cl_dt_release(void);       /* module cleanup_global */
#endif
