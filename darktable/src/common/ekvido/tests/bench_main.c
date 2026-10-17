/* Core benchmark: one case per process (peak RSS = that case). Prints
   "<case> <W>x<H> seconds=<s> maxrss_mb=<mb>". Cached cases run a warm-up call first and
   time the second, changed call — what an interactive slider move costs. */
#include "../energy.h"
#include "../color.h"
#include "../kazuki.h"
#include "../mist.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double maxrss_mb(void)
{
#ifdef _WIN32
  return 0.0; /* no getrusage on Windows: peak RSS not reported */
#else
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  return (double)ru.ru_maxrss / (1024.0 * 1024.0); /* bytes */
#else
  return (double)ru.ru_maxrss / 1024.0; /* KiB */
#endif
#endif
}

/* Smooth dim background, a few hundred coloured highlights, some clipped plateaus. */
static float *synthetic(int w, int h)
{
  const size_t np = (size_t)w * h;
  float *in = malloc(sizeof(float) * np * 4);
  if(!in) return NULL;
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      float *px = in + ((size_t)y * w + x) * 4;
      px[0] = 0.02f + 0.15f * (float)x / (float)w;
      px[1] = 0.03f + 0.10f * (float)y / (float)h;
      px[2] = 0.04f + 0.05f * (float)(x + y) / (float)(w + h);
      px[3] = 1.f;
    }
  uint32_t s = 2024u;
  for(int n = 0; n < 400; n++)
  {
    s = s * 1664525u + 1013904223u; const int cx = (int)(s % (uint32_t)w);
    s = s * 1664525u + 1013904223u; const int cy = (int)(s % (uint32_t)h);
    s = s * 1664525u + 1013904223u; const float v = 2.f + (float)(s % 3800u) / 100.f; /* 2..40 */
    const float tint[3][3] = { { 1.f, 0.6f, 0.25f }, { 0.4f, 0.7f, 1.f }, { 1.f, 1.f, 1.f } };
    const float *c = tint[n % 3];
    const int rad = n % 10 == 0 ? 6 : 2; /* every tenth a clipped plateau */
    for(int y = cy - rad; y <= cy + rad; y++)
      for(int x = cx - rad; x <= cx + rad; x++)
        if(x >= 0 && y >= 0 && x < w && y < h)
          for(int k = 0; k < 3; k++) in[((size_t)y * w + x) * 4 + k] = rad == 6 ? 60.f : v * c[k];
  }
  return in;
}

/* Module defaults (ekvido KazukiParams()), hues via the same HSV path as the module. */
static void kazuki_defaults(ekvido_kazuki_params_t *p)
{
  memset(p, 0, sizeof(*p));
  p->strength = 0.5f; p->blend = EKVIDO_BLEND_ADDITIVE; p->blend_white_ev = 0.f;
  p->threshold_ev = 0.f; p->knee_ev = 1.f;
  p->hue_radius = 0.01f; p->neutral_point = 0.f; p->warmth_softness = 0.05f; p->neutral_bias = 0.8f;
  p->push_warm = 0.7f; p->push_cool = 0.7f;
  ekvido_hsv_to_rgb(40.f, 0.45f, p->warm_core); ekvido_hsv_to_rgb(22.5f, 0.88f, p->warm_edge);
  ekvido_hsv_to_rgb(193.846154f, 0.65f, p->cool_core); ekvido_hsv_to_rgb(232.173913f, 0.92f, p->cool_edge);
  p->band_split = 0.006f;
  p->spec = (ekvido_kernel_spec_t){ 0.006f, 1.3f, 1.85f, 1.f, 0.1f, 1.2f, 4, 0.f, 0.001f, 0.96f, 0.32f, 2.93f,
                                    0.58f, 1.f, 0.f, 0.2f };
}

static void mist_defaults(ekvido_mist_params_t *p)
{
  memset(p, 0, sizeof(*p));
  p->strength = 0.15f;
  p->spec = (ekvido_kernel_spec_t){ 0.004f, 1.5f, 1.f, 1.f, 0.15f, 1.f, 0, 0.f, 0.001f, 1.3f, 0.3f, 1.f, 1.f,
                                    0.5f, 0.5f, 0.f };
}

int main(int argc, char **argv)
{
  if(argc != 4)
  {
    fprintf(stderr, "usage: %s kz-export|kz-shape|kz-colour|kz-strength|mist-export|mist-shape W H\n", argv[0]);
    return 2;
  }
  const char *c = argv[1];
  const int w = atoi(argv[2]), h = atoi(argv[3]);
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif
  const size_t np = (size_t)w * h;
  float *in = synthetic(w, h), *out = malloc(sizeof(float) * np * 4);
  if(!in || !out) { fprintf(stderr, "out of memory\n"); return 1; }
  const float diag = hypotf((float)w, (float)h);
  const ekvido_rect_t full = { 0, 0, w, h };
  double t0 = 0.0, t1 = 0.0;
  int err = 0;
  if(!strncmp(c, "kz-", 3))
  {
    ekvido_kazuki_params_t p;
    kazuki_defaults(&p);
    const float clip = ekvido_estimate_clip_level(in, np, 4);
    if(!strcmp(c, "kz-export"))
    {
      t0 = now();
      err = ekvido_kazuki_process(in, w, h, out, full, &p, diag, clip, 0, EKVIDO_VIEW_FINAL, NULL, nthreads);
      t1 = now();
    }
    else
    {
      ekvido_kazuki_cache_t *cache = ekvido_kazuki_cache_new();
      err = ekvido_kazuki_process(in, w, h, out, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, cache, nthreads);
      if(!strcmp(c, "kz-shape")) p.spec.halo_radius = 0.008f;
      else if(!strcmp(c, "kz-colour")) p.warm_edge[1] = 0.5f;
      else p.strength = 0.8f; /* kz-strength */
      t0 = now();
      err |= ekvido_kazuki_process(in, w, h, out, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, cache, nthreads);
      t1 = now();
      ekvido_kazuki_cache_free(cache);
    }
  }
  else
  {
    ekvido_mist_params_t p;
    mist_defaults(&p);
    if(!strcmp(c, "mist-export"))
    {
      t0 = now();
      err = ekvido_mist_process(in, w, h, out, full, &p, diag, 0, 0, NULL, nthreads);
      t1 = now();
    }
    else
    {
      ekvido_mist_cache_t *cache = ekvido_mist_cache_new();
      err = ekvido_mist_process(in, w, h, out, full, &p, diag, 1, 0, cache, nthreads);
      p.spec.halo_radius = 0.006f;
      t0 = now();
      err |= ekvido_mist_process(in, w, h, out, full, &p, diag, 1, 0, cache, nthreads);
      t1 = now();
      ekvido_mist_cache_free(cache);
    }
  }
  if(err) { fprintf(stderr, "%s: core returned %d\n", c, err); return 1; }
  printf("%s %dx%d seconds=%.3f maxrss_mb=%.0f\n", c, w, h, t1 - t0, maxrss_mb());
  free(in);
  free(out);
  return 0;
}
