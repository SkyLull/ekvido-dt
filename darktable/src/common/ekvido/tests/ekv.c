/* ekv.c — reader for the EKV1 fixtures written by ekvido/tools/make_c_fixtures.py */
#include "ekv.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ekv_read(const char *path, ekv_t *a)
{
  memset(a, 0, sizeof(*a));
  FILE *f = fopen(path, "rb");
  if(!f) { fprintf(stderr, "ekv: cannot open %s\n", path); return -1; }
  char magic[4];
  int32_t nd = 0;
  if(fread(magic, 1, 4, f) != 4 || memcmp(magic, "EKV1", 4) || fread(&nd, 4, 1, f) != 1 || nd < 1 || nd > 4)
  { fclose(f); fprintf(stderr, "ekv: bad header in %s\n", path); return -1; }
  a->ndim = nd;
  a->n = 1;
  for(int i = 0; i < nd; i++)
  {
    int32_t d = 0;
    if(fread(&d, 4, 1, f) != 1 || d < 0) { fclose(f); return -1; }
    a->dims[i] = d;
    a->n *= (size_t)d;
  }
  a->data = malloc(a->n * sizeof(float));
  const int ok = a->data && fread(a->data, sizeof(float), a->n, f) == a->n;
  fclose(f);
  if(!ok) { ekv_free(a); return -1; }
  return 0;
}

void ekv_free(ekv_t *a)
{
  free(a->data);
  a->data = NULL;
}
