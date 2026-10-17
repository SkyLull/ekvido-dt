#include "common/ekvido/hash.h"

uint64_t ekvido_hash_floats(const float *in, size_t n)
{
  enum { BLOCKS = 64 };
  uint64_t part[BLOCKS];
  const uint32_t *u = (const uint32_t *)in;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int b = 0; b < BLOCKS; b++)
  {
    uint64_t h = 1469598103934665603ULL;
    const size_t lo = n * b / BLOCKS, hi = n * (b + 1) / BLOCKS;
    for(size_t i = lo; i < hi; i++) { h ^= u[i]; h *= 1099511628211ULL; }
    part[b] = h;
  }
  uint64_t h = 1469598103934665603ULL ^ (uint64_t)n;
  for(int b = 0; b < BLOCKS; b++) { h ^= part[b]; h *= 1099511628211ULL; }
  return h;
}
