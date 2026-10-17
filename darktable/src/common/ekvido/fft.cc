// PocketFFT-backed FFT for the ekvido core (see fft.h).
#include "common/ekvido/fft.h"
#include "external/pocketfft/pocketfft_hdronly.h"

#include <complex>

extern "C" int ekvido_fft_size(int n)
{
  for(int m = n > 1 ? n : 1;; m++)
  {
    int k = m;
    for(const int p : { 2, 3, 5 })
      while(k % p == 0) k /= p;
    if(k == 1) return m;
  }
}

/* The 2-D transforms are PocketFFT's own two passes (rows, then columns), run here on blocks of
   CHUNK rows or columns, each block a single-threaded PocketFFT call, the blocks spread over
   OpenMP threads. PocketFFT's internal threading splits the rows at arbitrary points, and which
   rows take its SIMD path and which its scalar tail then depends on the thread count — under
   -ffast-math (darktable's release flags) those two paths round differently, so the bytes did too.
   CHUNK is a multiple of every PocketFFT SIMD width (VLEN ≤ 16), so every block but the last is all
   SIMD and the last one's tail is the same rows a single whole-array call ends on: the same bytes
   as one single-threaded call, for any thread count. */
namespace
{
typedef std::complex<float> cplx;
const size_t CHUNK = 64;

inline int threads(int nthreads) { return nthreads > 0 ? nthreads : 1; }

/* Columns [0, wc) of the h×wc half-spectrum, c2c along axis 0, from src to dst (may alias). */
void columns(const cplx *src, cplx *dst, size_t h, size_t wc, bool forward, int nthreads)
{
  const ptrdiff_t cs = (ptrdiff_t)(wc * sizeof(cplx));
  const long nb = (long)((wc + CHUNK - 1) / CHUNK);
  (void)nthreads;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads(nthreads))
#endif
  for(long b = 0; b < nb; b++)
  {
    const size_t c0 = (size_t)b * CHUNK, n = wc - c0 < CHUNK ? wc - c0 : CHUNK;
    pocketfft::c2c<float>({ h, n }, { cs, (ptrdiff_t)sizeof(cplx) }, { cs, (ptrdiff_t)sizeof(cplx) }, { 0 }, forward,
                          src + c0, dst + c0, 1.0f, 1);
  }
}
} // namespace

extern "C" void ekvido_rfft2(const float *in, float *spec, int h, int w, int nthreads)
{
  const size_t H = (size_t)h, W = (size_t)w, WC = W / 2 + 1;
  cplx *out = reinterpret_cast<cplx *>(spec);
  const ptrdiff_t rs = (ptrdiff_t)(W * sizeof(float)), cs = (ptrdiff_t)(WC * sizeof(cplx));
  const long nb = (long)((H + CHUNK - 1) / CHUNK);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads(nthreads))
#endif
  for(long b = 0; b < nb; b++)
  {
    const size_t r0 = (size_t)b * CHUNK, n = H - r0 < CHUNK ? H - r0 : CHUNK;
    pocketfft::r2c<float>({ n, W }, { rs, (ptrdiff_t)sizeof(float) }, { cs, (ptrdiff_t)sizeof(cplx) }, (size_t)1,
                          pocketfft::FORWARD, in + r0 * W, out + r0 * WC, 1.0f, 1);
  }
  columns(out, out, H, WC, pocketfft::FORWARD, nthreads);
}

extern "C" void ekvido_irfft2(const float *spec, float *out, int h, int w, int nthreads)
{
  const size_t H = (size_t)h, W = (size_t)w, WC = W / 2 + 1;
  const ptrdiff_t rs = (ptrdiff_t)(W * sizeof(float)), cs = (ptrdiff_t)(WC * sizeof(cplx));
  pocketfft::detail::arr<cplx> tmp(H * WC); /* spec stays untouched, as in PocketFFT's own c2r */
  columns(reinterpret_cast<const cplx *>(spec), tmp.data(), H, WC, pocketfft::BACKWARD, nthreads);
  const float fct = 1.0f / ((float)h * (float)w);
  const long nb = (long)((H + CHUNK - 1) / CHUNK);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads(nthreads))
#endif
  for(long b = 0; b < nb; b++)
  {
    const size_t r0 = (size_t)b * CHUNK, n = H - r0 < CHUNK ? H - r0 : CHUNK;
    pocketfft::c2r<float>({ n, W }, { cs, (ptrdiff_t)sizeof(cplx) }, { rs, (ptrdiff_t)sizeof(float) }, (size_t)1,
                          pocketfft::BACKWARD, tmp.data() + r0 * WC, out + r0 * W, fct, 1);
  }
}
