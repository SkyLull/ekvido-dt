#pragma once
/* Content hash for the per-pipe caches. */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* FNV-1a over 32-bit words in fixed blocks (parallel, deterministic), blocks combined in order. */
uint64_t ekvido_hash_floats(const float *in, size_t n);
#ifdef __cplusplus
}
#endif
