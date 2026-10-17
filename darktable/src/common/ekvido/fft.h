#pragma once
/* FFT boundary for the ekvido core. Everything FFT goes through here so a GPU FFT can
   replace PocketFFT later without touching the convolution code (spec §6.1). */
#ifdef __cplusplus
extern "C" {
#endif

int ekvido_fft_size(int n);
void ekvido_rfft2(const float *in, float *spec, int h, int w, int nthreads);
void ekvido_irfft2(const float *spec, float *out, int h, int w, int nthreads);

#ifdef __cplusplus
}
#endif
