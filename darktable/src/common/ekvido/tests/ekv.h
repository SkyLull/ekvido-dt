/* ekv.h */
#pragma once
#include <stddef.h>
typedef struct ekv_t { int ndim; int dims[4]; size_t n; float *data; } ekv_t;
int ekv_read(const char *path, ekv_t *a);   /* 0 ok, -1 on any error */
void ekv_free(ekv_t *a);
