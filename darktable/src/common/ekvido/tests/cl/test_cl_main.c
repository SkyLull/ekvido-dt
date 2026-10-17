/* GPU-vs-CPU tests of the ekvido OpenCL path (src/common/ekvido/cl): each stage against its C
   twin, then the pipelines against ekvido_kazuki_process_masked / ekvido_mist_process.
   Usage: ekvido_cl_tests <ekvido.cl> [bench W H]   (needs an OpenCL GPU) */
#include "common/ekvido/cl/ekv_cl_priv.h"
#include "common/ekvido/blend.h"
#include "common/ekvido/color.h"
#include "common/ekvido/convolve.h"
#include "common/ekvido/energy.h"
#include "common/ekvido/fft.h"
#include "common/ekvido/kazuki.h"
#include "common/ekvido/mist.h"
#include "common/ekvido/psf.h"
#include "common/ekvido/wblur.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#endif

void ekv_cl_use_linked_api(void);

static int g_fail = 0, g_checks = 0;
static cl_context ctx;
static cl_device_id dev;
static cl_command_queue q;
static ekv_cl_t *E;
static const char *g_source_path;

#define CHECK(cond, ...)                                                                    \
  do {                                                                                      \
    g_checks++;                                                                             \
    if(!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
                  fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); }                    \
  } while(0)

/* Same rule as the core tests: max |got - want| <= rel * max|want|; non-finite classes match. */
static int close_enough(const char *what, const float *got, const float *want, size_t n, double rel)
{
  double peak = 0.0, worst = 0.0;
  for(size_t i = 0; i < n; i++)
    if(isfinite(want[i])) peak = fmax(peak, fabs(want[i]));
  for(size_t i = 0; i < n; i++)
  {
    if(!isfinite(want[i]))
    {
      const int same = (isnan(want[i]) && isnan(got[i])) || (isinf(want[i]) && got[i] == want[i]);
      if(!same) { fprintf(stderr, "  %s[%zu]: want %g got %g\n", what, i, want[i], got[i]); return 0; }
      continue;
    }
    if(!isfinite(got[i])) { fprintf(stderr, "  %s[%zu]: non-finite %g\n", what, i, got[i]); return 0; }
    worst = fmax(worst, fabs((double)got[i] - want[i]));
  }
  const int ok = worst <= rel * peak + 1e-12;
  if(!ok) fprintf(stderr, "  %s: max error %.3g vs allowed %.3g (peak %.3g)\n", what, worst, rel * peak, peak);
  return ok;
}

/* The warmth view is an 8-bit quantised colormap (rint(v)/255): ulp-level differences in t flip single
   levels, so compare per pixel to one decoded level and bound how many channels differ at all. */
static int quantised_close(const char *what, const float *got, const float *want, size_t n)
{
  size_t diff = 0;
  double worst = 0.0;
  int ok = 1;
  for(size_t i = 0; i < n; i++)
  {
    if(!isfinite(want[i]))
    {
      ok &= (isnan(want[i]) && isnan(got[i])) || (isinf(want[i]) && got[i] == want[i]);
      continue;
    }
    if(!isfinite(got[i])) { ok = 0; continue; }
    const double d = fabs((double)got[i] - want[i]);
    worst = fmax(worst, d);
    if(d > 1e-5) diff++;
  }
  ok &= worst <= 0.01 && (double)diff <= 0.001 * (double)n;
  printf("  %s: %zu of %zu channels differ > 1e-5 (%.4f%%), max difference %.5f%s\n", what, diff, n,
         100.0 * (double)diff / (double)n, worst, ok ? "" : " FAIL");
  return ok;
}

__attribute__((unused)) static double now_ms(void)
{
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return 1e3 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
}

static cl_mem buf_from(const void *host, size_t bytes)
{
  cl_mem b = ekv_cl_buffer(E, bytes);
  if(b && host && ekv_cl_upload(E, b, host, bytes)) { ekv_cl_release(b); b = NULL; }
  return b;
}

static void buf_to_host(cl_mem b, void *host, size_t bytes)
{
  if(ekv_cl_download(E, b, host, bytes)) { fprintf(stderr, "download failed\n"); exit(1); }
}

static cl_mem img_new(int w, int h, const float *rgba)
{
  const cl_image_format fmt = { CL_RGBA, CL_FLOAT };
  cl_image_desc desc;
  memset(&desc, 0, sizeof(desc));
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = (size_t)w;
  desc.image_height = (size_t)h;
  cl_int e = CL_SUCCESS;
  cl_mem m = clCreateImage(ctx, CL_MEM_READ_WRITE | (rgba ? CL_MEM_COPY_HOST_PTR : 0), &fmt, &desc, (void *)rgba, &e);
  return e == CL_SUCCESS ? m : NULL;
}

static void img_read(cl_mem m, int w, int h, float *rgba)
{
  const size_t o[3] = { 0, 0, 0 }, r[3] = { (size_t)w, (size_t)h, 1 };
  if(clEnqueueReadImage(q, m, CL_TRUE, o, r, 0, 0, rgba, 0, NULL, NULL) != CL_SUCCESS)
  {
    fprintf(stderr, "image read failed\n");
    exit(1);
  }
}

static void lcg_fill(float *x, size_t n, uint32_t seed, float scale)
{
  uint32_t s = seed;
  for(size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; x[i] = scale * (float)(s >> 8) / 16777216.f; }
}

/* GPU forward vs PocketFFT and the round trip, on an fh×fw grid. */
static void test_fft(int fh, int fw)
{
  ekv_cl_grid_t g = { fh, fw, 0, fh, fw, 2 * (fw / 2 + 1) };
  const size_t real_n = (size_t)fh * fw, spec_n = ekv_cl_grid_bytes(&g) / sizeof(float);
  float *x = malloc(sizeof(float) * real_n), *want = malloc(sizeof(float) * spec_n);
  float *padded = calloc(spec_n, sizeof(float)), *got = malloc(sizeof(float) * spec_n);
  lcg_fill(x, real_n, 7u, 1.f);
  ekvido_rfft2(x, want, fh, fw, 1);
  for(int i = 0; i < fh; i++) memcpy(padded + (size_t)i * g.stride, x + (size_t)i * fw, sizeof(float) * fw);
  cl_mem b = buf_from(padded, sizeof(float) * spec_n);
  CHECK(b && !ekv_cl_fft(E, &g, b, 0), "fft %dx%d forward ran", fh, fw);
  buf_to_host(b, got, sizeof(float) * spec_n);
  CHECK(close_enough("fft forward", got, want, spec_n, 1e-5), "fft %dx%d forward vs PocketFFT", fh, fw);
  CHECK(!ekv_cl_fft(E, &g, b, 1), "fft %dx%d inverse ran", fh, fw);
  buf_to_host(b, got, sizeof(float) * spec_n);
  const float s = (float)(1.0 / ((double)fh * fw));
  for(int i = 0; i < fh; i++)
    for(int j = 0; j < fw; j++) padded[(size_t)i * fw + j] = got[(size_t)i * g.stride + j] * s;
  CHECK(close_enough("fft round trip", padded, x, real_n, 1e-5), "fft %dx%d round trip", fh, fw);
  for(int i = 0; i < EKV_PLANS; i++)
    if(E->plan[i].app && E->plan[i].fh == fh && E->plan[i].fw == fw)
    {
      printf("  plan %dx%d: %.0f ms (%s), VkFFT scratch %.0f MB, grid %.0f MB\n", fh, fw, E->plan[i].ms,
             E->plan[i].from_disk ? "disk cache" : "built", E->plan[i].temp / 1048576.0,
             ekv_cl_grid_bytes(&g) / 1048576.0);
      CHECK(E->plan[i].temp <= EKV_CL_FFT_TEMP_GRIDS * (uint64_t)ekv_cl_grid_bytes(&g),
            "fft %dx%d scratch within EKV_CL_FFT_TEMP_GRIDS", fh, fw);
      CHECK(E->fft_temp_bytes == E->plan[i].temp && (E->fft_temp != NULL) == (E->plan[i].temp != 0),
            "fft %dx%d scratch held only as E->fft_temp", fh, fw);
    }
  ekv_cl_fft_end(E);
  CHECK(!E->fft_temp && !E->fft_temp_bytes, "fft %dx%d scratch released by ekv_cl_fft_end", fh, fw);
  ekv_cl_release(b);
  free(x); free(want); free(padded); free(got);
}

static const ekv_cl_plan_t *find_plan(int fh, int fw)
{
  for(int i = 0; i < EKV_PLANS; i++)
    if(E->plan[i].app && E->plan[i].fh == fh && E->plan[i].fw == fw) return &E->plan[i];
  return NULL;
}

static long file_size(const char *path, uint64_t *first)
{
  FILE *f = fopen(path, "rb");
  long n = -1;
  if(!f) return n;
  if(first && fread(first, sizeof(*first), 1, f) != 1) *first = 0;
  if(!fseek(f, 0, SEEK_END)) n = ftell(f);
  fclose(f);
  return n;
}

/* Review Focus 5: a corrupt or truncated plan file is ignored, deleted and rebuilt; a good one is
   loaded in a fresh plan set. */
static void test_fft_plan_cache(void)
{
  if(!E->plan_dir) return;
  const int fh = 96, fw = 100;
  char path[1024];
  ekv_cl_fft_plan_path(E, fh, fw, path, sizeof(path));
  ekv_cl_fft_free_plans(E);
  FILE *f = fopen(path, "wb");
  if(f) { fputs("not a VkFFT plan", f); fclose(f); }
  test_fft(fh, fw);
  const ekv_cl_plan_t *p = find_plan(fh, fw);
  CHECK(p && !p->from_disk, "corrupt plan file not loaded (from_disk = %d)", p ? p->from_disk : -1);
  uint64_t first = 0;
  long n = file_size(path, &first);
  CHECK(n > 64 && first == (uint64_t)n, "corrupt plan file replaced by a real one (%ld bytes, header %llu)", n,
        (unsigned long long)first);
  ekv_cl_fft_free_plans(E);
  test_fft(fh, fw);
  p = find_plan(fh, fw);
  CHECK(p && p->from_disk == 1, "good plan file loaded from disk after free_plans");
  /* truncated: header says the full size */
  char *blob = malloc((size_t)n);
  f = fopen(path, "rb");
  const int got = f && blob && fread(blob, 1, (size_t)n, f) == (size_t)n;
  if(f) fclose(f);
  f = got ? fopen(path, "wb") : NULL;
  if(f) { fwrite(blob, 1, (size_t)n / 2, f); fclose(f); }
  free(blob);
  ekv_cl_fft_free_plans(E);
  test_fft(fh, fw);
  p = find_plan(fh, fw);
  CHECK(got && p && !p->from_disk, "truncated plan file not loaded");
  CHECK(file_size(path, &first) == n && first == (uint64_t)n, "truncated plan file rebuilt");
  /* a fresh device context finds the good file */
  char *src = ekv_cl_read_source(g_source_path);
  ekv_cl_t *E2 = src ? ekv_cl_new(ctx, dev, q, src, E->plan_dir) : NULL;
  free(src);
  ekv_cl_t *E1 = E;
  E = E2;
  if(E2) test_fft(fh, fw);
  p = E2 ? find_plan(fh, fw) : NULL;
  CHECK(p && p->from_disk == 1, "plan loaded from disk in a fresh ekv_cl_t");
  E = E1;
  ekv_cl_free(E2);
}

static int pick_device(void)
{
  cl_platform_id pl[8];
  cl_uint np = 0;
  if(clGetPlatformIDs(8, pl, &np) != CL_SUCCESS) return -1;
  for(cl_uint i = 0; i < np; i++)
    if(clGetDeviceIDs(pl[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) return 0;
  return -1;
}

/* Highlights of both temperatures, a clipped plateau (the image max, ~0.3 % of the frame), a
   gradient floor and, optionally, NaN / +Inf / -Inf / negative pixels. */
static void synth(float *px, int w, int h, int nonfinite)
{
  static const float spot[6][6] = { /* x, y (fractions), radius (fraction of w), r, g, b */
    { .2f, .3f, .02f, 30.f, 18.f, 6.f }, { .7f, .25f, .015f, 6.f, 14.f, 40.f },
    { .5f, .6f, .03f, 12.f, 12.f, 12.f }, { .85f, .8f, .01f, 50.f, 20.f, 4.f },
    { .1f, .85f, .02f, 4.f, 10.f, 25.f }, { .35f, .1f, .008f, 80.f, 60.f, 40.f } };
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
      float *p = px + ((size_t)y * w + x) * 4;
      const float base = 0.02f + 0.2f * (float)x / (float)w;
      p[0] = base * 1.1f; p[1] = base; p[2] = base * 0.9f; p[3] = 1.f;
      for(int s = 0; s < 6; s++)
      {
        const float dx = x - spot[s][0] * w, dy = y - spot[s][1] * h, r = fmaxf(spot[s][2] * w, 1.f);
        const float g = expf(-(dx * dx + dy * dy) / (2.f * r * r));
        for(int c = 0; c < 3; c++) p[c] += g * spot[s][3 + c];
      }
    }
  for(int y = (int)(h * 0.45f); y < (int)(h * 0.5f); y++)
    for(int x = (int)(w * 0.6f); x < (int)(w * 0.66f); x++)
      for(int c = 0; c < 3; c++) px[((size_t)y * w + x) * 4 + c] = 100.f;
  if(nonfinite)
  {
    px[((size_t)3 * w + 3) * 4 + 0] = NAN;
    px[((size_t)5 * w + 10) * 4 + 1] = INFINITY;
    px[((size_t)7 * w + 20) * 4 + 2] = -INFINITY;
    px[((size_t)9 * w + 40) * 4 + 0] = -5.f;
  }
}

static void planes_of(const float *rgba, size_t np, float *src)
{
  for(size_t i = 0; i < np; i++)
    for(int c = 0; c < 3; c++) src[c * np + i] = ekvido_positive(rgba[i * 4 + c]);
}

static void test_split_energy_mask(void)
{
  const int w = 257, h = 193;
  const size_t np = (size_t)w * h;
  float *px = malloc(sizeof(float) * np * 4), *src = malloc(sizeof(float) * np * 3);
  float *e = malloc(sizeof(float) * np * 3), *got = malloc(sizeof(float) * np * 3), *mask = malloc(sizeof(float) * np);
  synth(px, w, h, 1);
  planes_of(px, np, src);
  for(size_t i = 0; i < np; i++) mask[i] = (float)(i % 97) / 64.f - 0.2f; /* < 0, 0…1 and > 1 */
  mask[11] = NAN; mask[12] = INFINITY;
  cl_mem in = img_new(w, h, px), s = ekv_cl_buffer(E, sizeof(float) * np * 3), ed = ekv_cl_buffer(E, sizeof(float) * np * 3);
  CHECK(in && s && ed && !ekv_cl_split(E, in, w, h, s), "split ran");
  buf_to_host(s, got, sizeof(float) * np * 3);
  CHECK(!memcmp(got, src, sizeof(float) * np * 3), "split = positive(RGB) exactly");
  const float *s3[3] = { src, src + np, src + 2 * np };
  float *e3[3] = { e, e + np, e + 2 * np };
  for(int variant = 0; variant < 2; variant++) /* soft knee; hard threshold (knee 0) */
  {
    const float thr = variant ? 1.5f : 0.f, knee = variant ? 0.f : 1.f;
    ekvido_extract_energy(s3, np, thr, knee, e3);
    CHECK(!ekv_cl_energy(E, s, np, thr, knee, ed), "energy ran");
    buf_to_host(ed, got, sizeof(float) * np * 3);
    CHECK(close_enough("energy", got, e, np * 3, 1e-5), "energy thr %g knee %g", thr, knee);
  }
  cl_mem md = buf_from(mask, sizeof(float) * np);
  for(int inv = 0; inv < 2; inv++)
  {
    ekvido_extract_energy(s3, np, 0.f, 1.f, e3);
    ekvido_mask_energy(e3, mask, np, inv);
    CHECK(!ekv_cl_energy(E, s, np, 0.f, 1.f, ed) && !ekv_cl_mask_energy(E, ed, md, np, inv), "mask ran");
    buf_to_host(ed, got, sizeof(float) * np * 3);
    CHECK(close_enough("masked energy", got, e, np * 3, 1e-5), "mask energy invert %d", inv);
  }
  clReleaseMemObject(in); ekv_cl_release(s); ekv_cl_release(ed); ekv_cl_release(md);
  free(px); free(src); free(e); free(got); free(mask);
}

static void test_clip_level_and_hash(void)
{
  const int w = 257, h = 193;
  const size_t np = (size_t)w * h;
  float *px = malloc(sizeof(float) * np * 4);
  for(int variant = 0; variant < 3; variant++) /* plateau; no plateau; all black */
  {
    synth(px, w, h, 1);
    if(variant == 1)
      for(size_t i = 0; i < np * 4; i++) if(px[i] == 100.f) px[i] = 1.f;
    if(variant == 2)
      for(size_t i = 0; i < np * 4; i++) px[i] = (i % 4 == 3) ? 1.f : 0.f;
    const float want = ekvido_estimate_clip_level(px, np, 4);
    cl_mem in = img_new(w, h, px);
    const float got = ekv_cl_estimate_clip_level(E, in, w, h);
    CHECK(got == want, "clip level variant %d: got %g want %g", variant, got, want);
    clReleaseMemObject(in);
  }
  synth(px, w, h, 1);
  cl_mem a = img_new(w, h, px), b = img_new(w, h, px);
  px[((size_t)100 * w + 100) * 4 + 1] = nextafterf(px[((size_t)100 * w + 100) * 4 + 1], 1e30f);
  cl_mem c = img_new(w, h, px);
  int ea = 0, eb = 0, ec = 0;
  const uint64_t ha = ekv_cl_hash_image(E, a, w, h, &ea), hb = ekv_cl_hash_image(E, b, w, h, &eb), hc = ekv_cl_hash_image(E, c, w, h, &ec);
  CHECK(!ea && !eb && !ec && ha == hb, "hash is deterministic");
  CHECK(ha != hc, "hash sees a one-ulp change");
  clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(c);
  free(px);
}

static void test_blur(void)
{
  const int w = 257, h = 193;
  const size_t np = (size_t)w * h;
  float *x = malloc(sizeof(float) * np), *want = malloc(sizeof(float) * np), *got = malloc(sizeof(float) * np);
  lcg_fill(x, np, 11u, 3.f);
  const double sig[5] = { 0.0, 0.7, 2.5, 13.0, 40.0 }; /* copy, direct ×2, downsampled f = 3 and f = 10 */
  for(int s = 0; s < 5; s++)
  {
    ekvido_gaussian_blur(x, want, h, w, sig[s]);
    cl_mem in = buf_from(x, sizeof(float) * np), out = ekv_cl_buffer(E, sizeof(float) * np);
    CHECK(!ekv_cl_gaussian_blur(E, in, out, h, w, sig[s]), "blur sigma %g ran", sig[s]);
    buf_to_host(out, got, sizeof(float) * np);
    CHECK(close_enough("blur", got, want, np, 1e-6), "blur sigma %g", sig[s]);
    CHECK(!ekv_cl_gaussian_blur(E, in, in, h, w, sig[s]), "in-place blur ran");
    buf_to_host(in, got, sizeof(float) * np);
    CHECK(close_enough("blur in place", got, want, np, 1e-6), "in-place blur sigma %g", sig[s]);
    ekv_cl_release(in); ekv_cl_release(out);
  }
  free(x); free(want); free(got);
}

static void test_warmth(void)
{
  const int w = 257, h = 193;
  const size_t np = (size_t)w * h;
  float *px = malloc(sizeof(float) * np * 4), *src = malloc(sizeof(float) * np * 3);
  float *t = malloc(sizeof(float) * np), *tg = malloc(sizeof(float) * np);
  unsigned char *cl = malloc(np), *cg = malloc(np);
  synth(px, w, h, 1);
  planes_of(px, np, src);
  const float *s3[3] = { src, src + np, src + 2 * np };
  const float clip = ekvido_estimate_clip_level(px, np, 4);
  const double sig[2] = { 3.2, 16.0 }; /* hue_radius 0.01 and 0.05 of this frame: direct and downsampled */
  const float soft[2] = { 0.05f, 0.f };
  for(int v = 0; v < 2; v++)
  {
    CHECK(!ekvido_warmth_map(s3, h, w, clip, sig[v], 1.f, 0.f, soft[v], 0.8f, t, cl), "cpu warmth");
    cl_mem sd = buf_from(src, sizeof(float) * np * 3), td = ekv_cl_buffer(E, sizeof(float) * np), cd = ekv_cl_buffer(E, np);
    CHECK(!ekv_cl_warmth_map(E, sd, h, w, clip, sig[v], 1.f, 0.f, soft[v], 0.8f, td, cd), "gpu warmth ran");
    buf_to_host(td, tg, sizeof(float) * np);
    buf_to_host(cd, cg, np);
    CHECK(close_enough("warmth t", tg, t, np, 1e-5), "warmth t sigma %g softness %g", sig[v], soft[v]);
    CHECK(!memcmp(cg, cl, np), "clipped mask exact (sigma %g)", sig[v]);
    ekv_cl_release(sd); ekv_cl_release(td); ekv_cl_release(cd);
  }
  free(px); free(src); free(t); free(tg); free(cl); free(cg);
}

/* GPU plane → FFT → × kernel spectrum → inverse → crop vs ekvido_conv_apply; odd sizes, a pad
   bigger than the frame (Review Focus 4) and the 256-rounded GPU grid. */
static void test_grid_conv(void)
{
  const int sizes[3][3] = { { 193, 257, 40 }, { 17, 23, 30 }, { 600, 700, 120 } }; /* h, w, pad = r */
  for(int s = 0; s < 3; s++)
  {
    const int h = sizes[s][0], w = sizes[s][1], r = sizes[s][2], d = 2 * r + 1;
    const size_t np = (size_t)w * h;
    float *x = malloc(sizeof(float) * np), *ker = malloc(sizeof(float) * d * d);
    float *want = malloc(sizeof(float) * np), *got = malloc(sizeof(float) * np);
    lcg_fill(x, np, 3u + s, 5.f);
    lcg_fill(ker, (size_t)d * d, 9u + s, 1.f);
    double ks = 0.0;
    for(int i = 0; i < d * d; i++) ks += ker[i];
    for(int i = 0; i < d * d; i++) ker[i] = (float)(ker[i] / ks);
    const float *pl[1] = { x };
    ekvido_conv_t *cv = ekvido_conv_new(pl, 1, h, w, r, 1);
    float *outs[1] = { want };
    ekvido_conv_apply(cv, ker, r, outs);
    ekvido_conv_free(cv);
    const ekv_cl_grid_t g = ekv_cl_grid_make(h, w, r);
    cl_mem e = buf_from(x, sizeof(float) * np), kf = ekv_cl_buffer(E, ekv_cl_grid_bytes(&g));
    cl_mem grid = ekv_cl_buffer(E, ekv_cl_grid_bytes(&g)), out = ekv_cl_buffer(E, sizeof(float) * np);
    CHECK(!ekv_cl_grid_kernel(E, &g, ker, r, kf) && !ekv_cl_grid_basis(E, &g, e, e, 0, grid) && !ekv_cl_fft(E, &g, grid, 0)
              && !ekv_cl_cmul(E, &g, grid, kf) && !ekv_cl_fft(E, &g, grid, 1) && !ekv_cl_crop(E, &g, grid, out, 0),
          "grid conv %dx%d r %d ran (grid %dx%d)", w, h, r, g.fw, g.fh);
    buf_to_host(out, got, sizeof(float) * np);
    CHECK(close_enough("grid conv", got, want, np, 5e-5), "grid conv %dx%d r %d vs CPU", w, h, r);
    CHECK(ekv_cl_grid_kernel(E, &g, ker, r + 1, kf) == -1, "oversize kernel refused");
    ekv_cl_release(e); ekv_cl_release(kf); ekv_cl_release(grid); ekv_cl_release(out);
    free(x); free(ker); free(want); free(got);
  }
}

/* Basis planes into the grid (exact), and the MAC kernel against _mac_tile's arithmetic. */
static void test_basis_and_mac(void)
{
  const int w = 257, h = 193;
  const size_t np = (size_t)w * h;
  float *e = malloc(sizeof(float) * np * 3), *t = malloc(sizeof(float) * np), *plane = malloc(sizeof(float) * np);
  lcg_fill(e, np * 3, 21u, 4.f);
  lcg_fill(t, np, 22u, 1.f);
  const ekv_cl_grid_t g = ekv_cl_grid_make(h, w, 30);
  const size_t gf = ekv_cl_grid_bytes(&g) / sizeof(float), nc = ekv_cl_grid_bins(&g);
  float *gh = malloc(sizeof(float) * gf);
  cl_mem ed = buf_from(e, sizeof(float) * np * 3), td = buf_from(t, sizeof(float) * np), grid = ekv_cl_buffer(E, sizeof(float) * gf);
  const float *e3[3] = { e, e + np, e + 2 * np };
  for(int k = 0; k < EKVIDO_KAZUKI_NB; k++)
  {
    ekvido_kazuki_basis_plane(e3, t, np, k, plane);
    CHECK(!ekv_cl_grid_basis(E, &g, ed, td, k, grid), "basis %d ran", k);
    buf_to_host(grid, gh, sizeof(float) * gf);
    int ok = 1;
    for(int y = 0; y < g.fh && ok; y++)
      for(int x = 0; x < g.stride && ok; x++)
      {
        const float want = (y < h && x < w) ? plane[(size_t)y * w + x] : 0.f;
        ok = fabsf(gh[(size_t)y * g.stride + x] - want) <= 1e-6f * fabsf(want);
      }
    CHECK(ok, "basis plane %d in the padded grid", k);
  }
  /* MAC: random spectra, coefficients from real params, two planes accumulated */
  float *s = malloc(sizeof(float) * gf), *ki = malloc(sizeof(float) * gf), *ko = malloc(sizeof(float) * gf);
  float *acc = malloc(sizeof(float) * gf * 3), *got = malloc(sizeof(float) * gf);
  lcg_fill(s, gf, 31u, 2.f); lcg_fill(ki, gf, 32u, 1.f); lcg_fill(ko, gf, 33u, 1.f);
  ekvido_kazuki_params_t p;
  memset(&p, 0, sizeof(p));
  p.push_warm = 0.7f; p.push_cool = 0.6f;
  ekvido_hsv_to_rgb(40.f, 0.45f, p.warm_core); ekvido_hsv_to_rgb(22.5f, 0.88f, p.warm_edge);
  ekvido_hsv_to_rgb(193.8f, 0.65f, p.cool_core); ekvido_hsv_to_rgb(232.2f, 0.92f, p.cool_edge);
  float coef[2][3][EKVIDO_KAZUKI_NB];
  ekvido_kazuki_coefficients(&p, coef);
  cl_mem sd = buf_from(s, sizeof(float) * gf), kid = buf_from(ki, sizeof(float) * gf), kod = buf_from(ko, sizeof(float) * gf);
  const cl_mem ad[3] = { ekv_cl_buffer(E, sizeof(float) * gf), ekv_cl_buffer(E, sizeof(float) * gf), ekv_cl_buffer(E, sizeof(float) * gf) };
  for(int k = 0; k < 2; k++)
  {
    CHECK(!ekv_cl_mac(E, &g, sd, kid, kod, coef, k, ad), "mac %d ran", k);
    for(size_t m = 0; m < nc; m++) /* _mac_tile, verbatim */
    {
      const size_t qq = 2 * m;
      const float a = s[qq], b = s[qq + 1];
      const float ir = a * ki[qq] - b * ki[qq + 1], ii = a * ki[qq + 1] + b * ki[qq];
      const float orr = a * ko[qq] - b * ko[qq + 1], oi = a * ko[qq + 1] + b * ko[qq];
      for(int c = 0; c < 3; c++)
      {
        float *ac = acc + (size_t)c * gf;
        const float re = coef[0][c][k] * ir + coef[1][c][k] * orr, im = coef[0][c][k] * ii + coef[1][c][k] * oi;
        ac[qq] = k ? ac[qq] + re : re;
        ac[qq + 1] = k ? ac[qq + 1] + im : im;
      }
    }
  }
  for(int c = 0; c < 3; c++)
  {
    buf_to_host(ad[c], got, sizeof(float) * gf);
    CHECK(close_enough("mac", got, acc + (size_t)c * gf, 2 * nc, 1e-6), "mac accumulator %d", c);
  }
  for(int c = 0; c < 3; c++) ekv_cl_release(ad[c]);
  ekv_cl_release(sd); ekv_cl_release(kid); ekv_cl_release(kod); ekv_cl_release(ed); ekv_cl_release(td); ekv_cl_release(grid);
  free(e); free(t); free(plane); free(gh); free(s); free(ki); free(ko); free(acc); free(got);
}

/* Output stage: every kazuki view and blend mode, mist mix at k < 1 and k = 1, with non-finite
   input pixels (Review Focus 2) and an off-centre rect. */
static void test_outputs(void)
{
  const int iw = 257, ih = 193;
  const size_t np = (size_t)iw * ih;
  const ekvido_rect_t rect = { 11, 7, 200, 150 };
  const size_t no = (size_t)rect.w * rect.h * 4;
  float *px = malloc(sizeof(float) * np * 4), *glow = malloc(sizeof(float) * np * 3), *en = malloc(sizeof(float) * np * 3);
  float *t = malloc(sizeof(float) * np), *want = malloc(sizeof(float) * no), *got = malloc(sizeof(float) * no);
  unsigned char *cl = malloc(np);
  synth(px, iw, ih, 1);
  for(int i = 0; i < 4; i++) /* move the non-finite pixels into the rect */
    px[((size_t)(rect.y + 2 + i) * iw + rect.x + 3 + i) * 4 + i % 3] = i == 0 ? NAN : (i == 1 ? INFINITY : (i == 2 ? -INFINITY : -2.f));
  lcg_fill(glow, np * 3, 41u, 3.f);
  lcg_fill(en, np * 3, 42u, 2.f);
  lcg_fill(t, np, 43u, 1.2f); /* also outside 0…1: the colormap clamps */
  for(size_t i = 0; i < np; i++) cl[i] = (unsigned char)(i % 7 == 0);
  for(size_t i = 0; i < np; i += 13) /* glow-free pixels for overlay (glow only) */
    for(int c = 0; c < 3; c++) glow[c * np + i] = 0.f;
  cl_mem in = img_new(iw, ih, px), out = img_new(rect.w, rect.h, NULL);
  cl_mem gd = buf_from(glow, sizeof(float) * np * 3), edd = buf_from(en, sizeof(float) * np * 3);
  cl_mem tdd = buf_from(t, sizeof(float) * np), cd = buf_from(cl, np);
  const float white = 1.5f;
  for(int view = 0; view <= EKVIDO_VIEW_CLIPPED; view++)
    for(int mode = 0; mode <= EKVIDO_BLEND_OVERLAY_GLOW; mode++)
    {
      if(view != EKVIDO_VIEW_FINAL && mode) continue;
      const float strength = mode == EKVIDO_BLEND_ADDITIVE ? 1.3f : 0.6f;
      for(int y = 0; y < rect.h; y++)
        for(int x = 0; x < rect.w; x++)
        {
          const size_t si = (size_t)(rect.y + y) * iw + (rect.x + x);
          const float *s = px + si * 4;
          float *o = want + ((size_t)y * rect.w + x) * 4;
          switch(view)
          {
            case EKVIDO_VIEW_GLOW: for(int c = 0; c < 3; c++) o[c] = glow[c * np + si]; break;
            case EKVIDO_VIEW_ENERGY: for(int c = 0; c < 3; c++) o[c] = en[c * np + si]; break;
            case EKVIDO_VIEW_WARMTH: ekvido_warmth_colour(t[si], o); break;
            case EKVIDO_VIEW_CLIPPED: for(int c = 0; c < 3; c++) o[c] = cl[si] ? (c == 0 ? 1.f : 0.f) : s[c]; break;
            default:
            {
              const float gl[3] = { glow[si], glow[np + si], glow[2 * np + si] };
              const float e[3] = { en[si], en[np + si], en[2 * np + si] };
              ekvido_blend_px((ekvido_blend_mode_t)mode, s, gl, e, strength, white, o);
            }
          }
          o[3] = s[3];
        }
      CHECK(!ekv_cl_kazuki_out(E, in, out, rect, iw, ih, (ekvido_kazuki_view_t)view, gd, edd, tdd, cd,
                               (ekvido_blend_mode_t)mode, strength, white), "kazuki out view %d mode %d ran", view, mode);
      img_read(out, rect.w, rect.h, got);
      CHECK(close_enough("kazuki out", got, want, no, 1e-5), "kazuki out view %d mode %d", view, mode);
    }
  { /* overlay (glow only) at strength 0, with glow-free pixels in the frame */
    for(int y = 0; y < rect.h; y++)
      for(int x = 0; x < rect.w; x++)
      {
        const size_t si = (size_t)(rect.y + y) * iw + (rect.x + x);
        const float *s = px + si * 4;
        float *o = want + ((size_t)y * rect.w + x) * 4;
        const float gl[3] = { glow[si], glow[np + si], glow[2 * np + si] };
        const float e[3] = { en[si], en[np + si], en[2 * np + si] };
        ekvido_blend_px(EKVIDO_BLEND_OVERLAY_GLOW, s, gl, e, 0.f, white, o);
        o[3] = s[3];
      }
    CHECK(!ekv_cl_kazuki_out(E, in, out, rect, iw, ih, EKVIDO_VIEW_FINAL, gd, edd, tdd, cd, EKVIDO_BLEND_OVERLAY_GLOW, 0.f,
                             white), "kazuki out overlay_glow strength 0 ran");
    img_read(out, rect.w, rect.h, got);
    CHECK(close_enough("kazuki out", got, want, no, 1e-5), "kazuki out overlay_glow strength 0");
  }
  const float ks[2] = { 0.15f, 1.f };
  for(int glow_only = 0; glow_only < 2; glow_only++)
    for(int ki = 0; ki < 2; ki++)
    {
      const float k = ks[ki];
      for(int y = 0; y < rect.h; y++)
        for(int x = 0; x < rect.w; x++)
        {
          const size_t si = (size_t)(rect.y + y) * iw + (rect.x + x);
          const float *s = px + si * 4;
          float *o = want + ((size_t)y * rect.w + x) * 4;
          for(int c = 0; c < 3; c++)
          {
            const float gg = glow[c * np + si];
            o[c] = glow_only ? gg : isfinite(s[c]) ? (1.f - k) * s[c] + k * gg : (k < 1.f ? s[c] : NAN);
          }
          o[3] = s[3];
        }
      CHECK(!ekv_cl_mist_out(E, in, out, rect, iw, ih, gd, k, glow_only), "mist out ran");
      img_read(out, rect.w, rect.h, got);
      CHECK(close_enough("mist out", got, want, no, 1e-5), "mist out k %g glow_only %d", k, glow_only);
    }
  CHECK(!ekv_cl_copy_rect(E, in, out, rect), "copy rect ran");
  img_read(out, rect.w, rect.h, got);
  int same = 1;
  for(int y = 0; y < rect.h; y++)
    same &= !memcmp(got + (size_t)y * rect.w * 4, px + ((size_t)(rect.y + y) * iw + rect.x) * 4, sizeof(float) * rect.w * 4);
  CHECK(same, "copy rect exact");
  /* coverage division */
  float *cover = malloc(sizeof(float) * np), *g2 = malloc(sizeof(float) * np * 3);
  lcg_fill(cover, np, 44u, 1.f);
  cover[5] = 0.f; /* floors at 1e-6 */
  for(size_t i = 0; i < np; i++)
  {
    const float inv = 1.f / fmaxf(cover[i], 1e-6f);
    for(int c = 0; c < 3; c++) g2[c * np + i] = glow[c * np + i] * inv;
  }
  cl_mem cvd = buf_from(cover, sizeof(float) * np);
  CHECK(!ekv_cl_div_cover(E, gd, cvd, np), "div cover ran");
  buf_to_host(gd, glow, sizeof(float) * np * 3);
  CHECK(close_enough("div cover", glow, g2, np * 3, 1e-6), "coverage division");
  clReleaseMemObject(in); clReleaseMemObject(out);
  ekv_cl_release(gd); ekv_cl_release(edd); ekv_cl_release(tdd); ekv_cl_release(cd); ekv_cl_release(cvd);
  free(px); free(glow); free(en); free(t); free(want); free(got); free(cl); free(cover); free(g2);
}

/* The module's defaults (src/iop/ekvidokazuki.c _defaults) as core params. */
static void kz_defaults(ekvido_kazuki_params_t *k)
{
  memset(k, 0, sizeof(*k));
  k->strength = 0.5f; k->blend = EKVIDO_BLEND_ADDITIVE; k->blend_white_ev = 0.f;
  k->threshold_ev = 0.f; k->knee_ev = 1.f; k->hue_radius = 0.01f; k->neutral_point = 0.f;
  k->warmth_softness = 0.05f; k->neutral_bias = 0.8f; k->push_warm = 0.7f; k->push_cool = 0.7f;
  ekvido_hsv_to_rgb(40.f, 0.45f, k->warm_core);
  ekvido_hsv_to_rgb(22.5f, 0.88f, k->warm_edge);
  ekvido_hsv_to_rgb(193.846154f, 0.65f, k->cool_core);
  ekvido_hsv_to_rgb(232.173913f, 0.92f, k->cool_edge);
  k->band_split = 0.006f;
  k->spec = (ekvido_kernel_spec_t){ .halo_radius = 0.006f, .halo_beta = 1.3f, .halo_stretch_x = 1.85f,
                                    .halo_stretch_y = 1.f, .veil_radius = 0.1f, .veil_beta = 1.2f, .arms = 4,
                                    .arm_angle = 0.f, .arm_radius = 0.001f, .arm_beta = 0.96f,
                                    .arm_sharpness = 0.32f, .arm_stretch_x = 2.93f, .arm_stretch_y = 0.58f,
                                    .w_halo = 1.f, .w_veil = 0.f, .w_arms = 0.2f };
}

/* src/iop/ekvidomist.c defaults (no star). */
static void mist_defaults(ekvido_mist_params_t *m)
{
  memset(m, 0, sizeof(*m));
  m->strength = 0.15f;
  m->spec = (ekvido_kernel_spec_t){ .halo_radius = 0.004f, .halo_beta = 1.5f, .halo_stretch_x = 1.f,
                                    .halo_stretch_y = 1.f, .veil_radius = 0.15f, .veil_beta = 1.f, .arms = 0,
                                    .w_halo = 0.5f, .w_veil = 0.5f, .w_arms = 0.f };
}

/* One GPU kazuki run into a host rect; returns the error code. */
static int gpu_kazuki(const float *px, int iw, int ih, ekvido_rect_t rect, const ekvido_kazuki_params_t *k, float diag,
                      float clip, const float *mask, int inv, int pad_to_cap, ekvido_kazuki_view_t view,
                      ekv_cl_cache_t *cache, float *out)
{
  cl_mem in = img_new(iw, ih, px), o = img_new(rect.w, rect.h, NULL);
  const int err = ekv_cl_kazuki_process(E, in, iw, ih, o, rect, k, diag, clip, mask, inv, pad_to_cap, view, cache);
  if(!err) img_read(o, rect.w, rect.h, out);
  clReleaseMemObject(in);
  clReleaseMemObject(o);
  return err;
}

static int gpu_mist(const float *px, int iw, int ih, ekvido_rect_t rect, const ekvido_mist_params_t *m, float diag,
                    int pad_to_cap, int glow_only, ekv_cl_cache_t *cache, float *out)
{
  cl_mem in = img_new(iw, ih, px), o = img_new(rect.w, rect.h, NULL);
  const int err = ekv_cl_mist_process(E, in, iw, ih, o, rect, m, diag, pad_to_cap, glow_only, cache);
  if(!err) img_read(o, rect.w, rect.h, out);
  clReleaseMemObject(in);
  clReleaseMemObject(o);
  return err;
}

/* GPU vs CPU end to end: views, blend modes, source mask (plain/inverted), pad modes, the
   downsampled warmth blur, non-finite pixels. */
static void test_kazuki_pipeline(void)
{
  const int iw = 257, ih = 193;
  const size_t np = (size_t)iw * ih;
  const ekvido_rect_t rect = { 9, 6, 230, 170 };
  const size_t no = (size_t)rect.w * rect.h * 4;
  const float diag = hypotf((float)iw, (float)ih);
  float *px = malloc(sizeof(float) * np * 4), *mask = malloc(sizeof(float) * np);
  float *want = malloc(sizeof(float) * no), *got = malloc(sizeof(float) * no);
  synth(px, iw, ih, 1);
  for(size_t i = 0; i < np; i++) mask[i] = (float)(i % (size_t)iw) / (float)iw;
  mask[17] = NAN;
  const float clip = ekvido_estimate_clip_level(px, np, 4);
  struct { const char *name; int view, blend, use_mask, inv, pad_to_cap; float hue_radius; } cases[] = {
    { "default", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_ADDITIVE, 0, 0, 0, 0.01f },
    { "cap", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_ADDITIVE, 0, 0, 1, 0.01f },
    { "conserve", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_CONSERVE, 0, 0, 0, 0.01f },
    { "screen", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_SCREEN, 0, 0, 0, 0.01f },
    { "overlay_glow", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_OVERLAY_GLOW, 0, 0, 0, 0.01f },
    { "glow_dodge wide", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_GLOW_DODGE, 0, 0, 0, 0.05f },
    { "mask", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_ADDITIVE, 1, 0, 0, 0.01f },
    { "mask inverted conserve", EKVIDO_VIEW_FINAL, EKVIDO_BLEND_CONSERVE, 1, 1, 0, 0.01f },
    { "view glow", EKVIDO_VIEW_GLOW, 0, 0, 0, 0, 0.01f },
    { "view energy", EKVIDO_VIEW_ENERGY, 0, 1, 0, 0, 0.01f },
    { "view warmth", EKVIDO_VIEW_WARMTH, 0, 0, 0, 0, 0.05f },
    { "view clipped", EKVIDO_VIEW_CLIPPED, 0, 0, 0, 0, 0.01f },
  };
  for(size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
  {
    ekvido_kazuki_params_t k;
    kz_defaults(&k);
    k.blend = (ekvido_blend_mode_t)cases[c].blend;
    k.hue_radius = cases[c].hue_radius;
    const float *m = cases[c].use_mask ? mask : NULL;
    CHECK(!ekvido_kazuki_process_masked(px, iw, ih, want, rect, &k, diag, clip, m, cases[c].inv, cases[c].pad_to_cap,
                                        (ekvido_kazuki_view_t)cases[c].view, NULL, 1), "cpu %s", cases[c].name);
    CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, clip, m, cases[c].inv, cases[c].pad_to_cap,
                      (ekvido_kazuki_view_t)cases[c].view, NULL, got), "gpu %s ran", cases[c].name);
    if(cases[c].view == EKVIDO_VIEW_WARMTH)
      CHECK(quantised_close(cases[c].name, got, want, no), "kazuki %s: GPU vs CPU (quantised)", cases[c].name);
    else
      CHECK(close_enough(cases[c].name, got, want, no, 5e-5), "kazuki %s: GPU vs CPU", cases[c].name);
  }
  /* strength 0: exact copy */
  ekvido_kazuki_params_t k;
  kz_defaults(&k);
  k.strength = 0.f;
  CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, clip, NULL, 0, 0, EKVIDO_VIEW_FINAL, NULL, got), "strength 0 ran");
  int same = 1;
  for(int y = 0; y < rect.h; y++)
    same &= !memcmp(got + (size_t)y * rect.w * 4, px + ((size_t)(rect.y + y) * iw + rect.x) * 4, sizeof(float) * rect.w * 4);
  CHECK(same, "kazuki strength 0 = input");
  free(px); free(mask); free(want); free(got);
}

/* Review Focus 3: what hits and what misses; cached output = uncached output, byte for byte. */
static void test_kazuki_cache(void)
{
  const int iw = 257, ih = 193;
  const size_t np = (size_t)iw * ih;
  const ekvido_rect_t rect = { 0, 0, iw, ih };
  const size_t no = np * 4;
  const float diag = hypotf((float)iw, (float)ih);
  float *px = malloc(sizeof(float) * no), *mask = malloc(sizeof(float) * np);
  float *a = malloc(sizeof(float) * no), *b = malloc(sizeof(float) * no);
  synth(px, iw, ih, 1);
  for(size_t i = 0; i < np; i++) mask[i] = (float)(i % 50) / 49.f;
  float clip = ekvido_estimate_clip_level(px, np, 4);
  ekv_cl_cache_t *c = ekv_cl_cache_new();
  ekvido_kazuki_params_t k;
  kz_defaults(&k);
  int hits = 0;
#define STEP(label, expect_hit, MASK, INV, VIEW)                                                                \
  do {                                                                                                          \
    CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, clip, MASK, INV, 1, VIEW, c, a), "%s cached ran", label);     \
    CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, clip, MASK, INV, 1, VIEW, NULL, b), "%s uncached ran", label); \
    CHECK(!memcmp(a, b, sizeof(float) * no), "%s: cached = uncached", label);                                   \
    const int h2 = ekv_cl_cache_hits(c);                                                                        \
    CHECK((h2 - hits) == (expect_hit), "%s: %s (hits %d -> %d)", label, (expect_hit) ? "hit" : "miss", hits, h2); \
    hits = h2;                                                                                                  \
  } while(0)
  STEP("first run", 0, NULL, 0, EKVIDO_VIEW_FINAL);
  k.strength = 0.8f;
  STEP("strength", 1, NULL, 0, EKVIDO_VIEW_FINAL);
  k.blend = EKVIDO_BLEND_SCREEN; k.blend_white_ev = 1.f;
  STEP("blend mode + white", 1, NULL, 0, EKVIDO_VIEW_FINAL);
  STEP("glow view", 1, NULL, 0, EKVIDO_VIEW_GLOW);
  STEP("energy view", 0, NULL, 0, EKVIDO_VIEW_ENERGY);
  STEP("back to final after a preview", 1, NULL, 0, EKVIDO_VIEW_FINAL);
  k.push_warm = 0.5f;
  STEP("colour", 0, NULL, 0, EKVIDO_VIEW_FINAL);
  k.spec.halo_radius = 0.01f;
  STEP("shape", 0, NULL, 0, EKVIDO_VIEW_FINAL);
  STEP("mask", 0, mask, 0, EKVIDO_VIEW_FINAL);
  STEP("mask inverted", 0, mask, 1, EKVIDO_VIEW_FINAL);
  px[((size_t)50 * iw + 60) * 4] += 1.f;
  STEP("input", 0, mask, 1, EKVIDO_VIEW_FINAL);
  k.strength = 0.6f;
  STEP("strength with mask", 1, mask, 1, EKVIDO_VIEW_FINAL);
  clip *= 0.9f;
  STEP("clip level", 0, mask, 1, EKVIDO_VIEW_FINAL);
#undef STEP
  /* Review Focus 1: a run that fails at any device allocation returns -2, leaks nothing it
     can see, and leaves no glow cached — the next run misses and matches an uncached run. */
  int failures = 0;
  for(int at = 1; at <= 40; at++)
  {
    k.push_cool = 0.3f + 0.01f * at; /* a colour change: always a miss, the full pipeline runs */
    ekv_cl_fail_alloc_at = at;
    const int err = gpu_kazuki(px, iw, ih, rect, &k, diag, clip, mask, 1, 1, EKVIDO_VIEW_FINAL, c, a);
    const int failed_inside = ekv_cl_fail_alloc_at == 0; /* the counter reached the pipeline */
    ekv_cl_fail_alloc_at = 0;
    if(!failed_inside) break; /* past the last allocation of a run */
    failures++;
    CHECK(err == -2, "allocation %d failing: run reports -2 (got %d)", at, err);
    const int before = ekv_cl_cache_hits(c);
    CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, clip, mask, 1, 1, EKVIDO_VIEW_FINAL, c, a), "rerun after failure %d", at);
    CHECK(ekv_cl_cache_hits(c) == before, "after failure %d nothing was cached (rerun missed)", at);
    CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, clip, mask, 1, 1, EKVIDO_VIEW_FINAL, NULL, b), "uncached %d", at);
    CHECK(!memcmp(a, b, sizeof(float) * no), "rerun after failure %d = uncached", at);
  }
  CHECK(failures >= 10, "fault injection reached %d allocation points", failures);
  ekv_cl_cache_free(c);
  free(px); free(mask); free(a); free(b);
}

static void test_mist_pipeline_and_cache(void)
{
  const int iw = 257, ih = 193;
  const size_t np = (size_t)iw * ih;
  const ekvido_rect_t rect = { 5, 4, 240, 180 };
  const size_t no = (size_t)rect.w * rect.h * 4;
  const float diag = hypotf((float)iw, (float)ih);
  float *px = malloc(sizeof(float) * np * 4), *want = malloc(sizeof(float) * no), *got = malloc(sizeof(float) * no);
  float *b = malloc(sizeof(float) * no);
  synth(px, iw, ih, 1);
  ekvido_mist_params_t m;
  const float ks[3] = { 0.15f, 1.f, 0.f };
  for(int i = 0; i < 3; i++)
    for(int glow_only = 0; glow_only < 2; glow_only++)
      for(int cap = 0; cap < 2; cap++)
      {
        mist_defaults(&m);
        m.strength = ks[i];
        CHECK(!ekvido_mist_process(px, iw, ih, want, rect, &m, diag, cap, glow_only, NULL, 1), "cpu mist");
        CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, cap, glow_only, NULL, got), "gpu mist ran");
        CHECK(close_enough("mist", got, want, no, 5e-5), "mist k %g glow_only %d cap %d", ks[i], glow_only, cap);
      }
  ekv_cl_cache_t *c = ekv_cl_cache_new();
  mist_defaults(&m);
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, c, got) && ekv_cl_cache_hits(c) == 0, "mist first run misses");
  m.strength = 0.4f;
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, c, got) && ekv_cl_cache_hits(c) == 1, "mist strength hits");
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, NULL, b) && !memcmp(got, b, sizeof(float) * no), "mist cached = uncached");
  m.spec.veil_radius = 0.08f;
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, c, got) && ekv_cl_cache_hits(c) == 1, "mist shape misses");
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, NULL, b) && !memcmp(got, b, sizeof(float) * no),
        "mist shape: cached = uncached");
  px[((size_t)50 * iw + 60) * 4] += 1.f;
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, c, got) && ekv_cl_cache_hits(c) == 1, "mist input change misses");
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, NULL, b) && !memcmp(got, b, sizeof(float) * no),
        "mist input change: cached = uncached");
  ekv_cl_cache_free(c);
  free(px); free(want); free(got); free(b);
}

/* Review Focus 4 / 5 of Phase 1 on the GPU: frame smaller than the kernel; nothing above threshold. */
static void test_tiny_and_dark(void)
{
  const int iw = 23, ih = 17;
  const size_t np = (size_t)iw * ih, no = np * 4;
  const ekvido_rect_t rect = { 0, 0, iw, ih };
  const float diag = 400.f; /* glow sized for a bigger frame: pad > frame */
  float *px = malloc(sizeof(float) * no), *want = malloc(sizeof(float) * no), *got = malloc(sizeof(float) * no);
  synth(px, iw, ih, 0);
  ekvido_kazuki_params_t k;
  kz_defaults(&k);
  const float clip = ekvido_estimate_clip_level(px, np, 4);
  CHECK(!ekvido_kazuki_process_masked(px, iw, ih, want, rect, &k, diag, clip, NULL, 0, 1, EKVIDO_VIEW_FINAL, NULL, 1), "cpu tiny");
  CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, clip, NULL, 0, 1, EKVIDO_VIEW_FINAL, NULL, got), "gpu tiny ran");
  CHECK(close_enough("tiny kazuki", got, want, no, 5e-5), "tiny kazuki GPU vs CPU");
  ekvido_mist_params_t m;
  mist_defaults(&m);
  CHECK(!ekvido_mist_process(px, iw, ih, want, rect, &m, diag, 1, 0, NULL, 1), "cpu tiny mist");
  CHECK(!gpu_mist(px, iw, ih, rect, &m, diag, 1, 0, NULL, got), "gpu tiny mist ran");
  CHECK(close_enough("tiny mist", got, want, no, 5e-5), "tiny mist GPU vs CPU");
  for(size_t i = 0; i < no; i++) px[i] = (i % 4 == 3) ? 1.f : 0.01f; /* everything far below 0 EV */
  CHECK(!gpu_kazuki(px, iw, ih, rect, &k, diag, INFINITY, NULL, 0, 1, EKVIDO_VIEW_FINAL, NULL, got), "dark ran");
  CHECK(!memcmp(got, px, sizeof(float) * no), "no energy: additive output = input exactly");
  free(px); free(want); free(got);
}

/* GPU core timings on a synthetic frame (cf. packaging/ekvido/bench-core.sh): each case after one
   warm-up run, so plan creation is excluded and reported separately. */
static void bench(int w, int h)
{
  if(w <= 0 || h <= 0) { w = 6016; h = 4016; }
  const size_t np = (size_t)w * h;
  float *px = malloc(sizeof(float) * np * 4);
  if(!px) return;
  synth(px, w, h, 0);
  const float diag = hypotf((float)w, (float)h);
  const ekvido_rect_t rect = { 0, 0, w, h };
  cl_mem in = img_new(w, h, px), out = img_new(w, h, NULL);
  ekvido_kazuki_params_t k;
  kz_defaults(&k);
  ekvido_mist_params_t m;
  mist_defaults(&m);
  const float clip = ekv_cl_estimate_clip_level(E, in, w, h);
  ekv_cl_cache_t *c = ekv_cl_cache_new();
  double t0 = now_ms();
  int err = ekv_cl_kazuki_process(E, in, w, h, out, rect, &k, diag, clip, NULL, 0, 0, EKVIDO_VIEW_FINAL, NULL);
  clFinish(q);
  printf("bench %dx%d first kazuki export (incl. plans): %.0f ms%s\n", w, h, now_ms() - t0, err ? " ERROR" : "");
#define TIME(label, call)                                                                    \
  do {                                                                                       \
    t0 = now_ms();                                                                           \
    err = (call);                                                                            \
    clFinish(q);                                                                             \
    printf("bench %dx%d %s: %.0f ms%s\n", w, h, label, now_ms() - t0, err ? " ERROR" : "");  \
  } while(0)
  TIME("clip level", isnan(ekv_cl_estimate_clip_level(E, in, w, h)));
  TIME("kz-export", ekv_cl_kazuki_process(E, in, w, h, out, rect, &k, diag, clip, NULL, 0, 0, EKVIDO_VIEW_FINAL, NULL));
  TIME("kz-interactive first (cap grid)", ekv_cl_kazuki_process(E, in, w, h, out, rect, &k, diag, clip, NULL, 0, 1, EKVIDO_VIEW_FINAL, c));
  k.strength = 0.8f;
  TIME("kz-strength (cache hit)", ekv_cl_kazuki_process(E, in, w, h, out, rect, &k, diag, clip, NULL, 0, 1, EKVIDO_VIEW_FINAL, c));
  k.push_warm = 0.5f;
  TIME("kz-colour", ekv_cl_kazuki_process(E, in, w, h, out, rect, &k, diag, clip, NULL, 0, 1, EKVIDO_VIEW_FINAL, c));
  k.spec.halo_radius = 0.01f;
  TIME("kz-shape", ekv_cl_kazuki_process(E, in, w, h, out, rect, &k, diag, clip, NULL, 0, 1, EKVIDO_VIEW_FINAL, c));
  err = ekv_cl_mist_process(E, in, w, h, out, rect, &m, diag, 0, 0, NULL); /* warm-up: mist grid plan */
  clFinish(q);
  TIME("mist-export", ekv_cl_mist_process(E, in, w, h, out, rect, &m, diag, 0, 0, NULL));
#undef TIME
  printf("bench %dx%d FFT scratch held after the runs: %s\n", w, h, E->fft_temp ? "YES (bug)" : "none");
  printf("bench %dx%d device bytes: kazuki %.0f MB, mist %.0f MB (estimates, images excluded)\n", w, h,
         ekv_cl_kazuki_bytes(w, h, ekvido_kernel_half_size(&k.spec, diag)) / 1048576.0,
         ekv_cl_mist_bytes(w, h, ekvido_kernel_half_size(&m.spec, diag)) / 1048576.0);
  ekv_cl_cache_free(c);
  clReleaseMemObject(in);
  clReleaseMemObject(out);
  free(px);
}

int main(int argc, char **argv)
{
  if(argc < 2) { fprintf(stderr, "usage: %s <ekvido.cl> [bench W H]\n", argv[0]); return 2; }
  g_source_path = argv[1];
  ekv_cl_use_linked_api();
  if(pick_device()) { fprintf(stderr, "no OpenCL GPU\n"); return 1; }
  char name[256] = { 0 };
  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
  cl_int e = CL_SUCCESS;
  ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
  q = clCreateCommandQueue(ctx, dev, 0, &e);
  char *src = ekv_cl_read_source(argv[1]);
#ifdef _WIN32
  /* system("mkdir -p") would run in cmd.exe: create the directories natively */
  const char *home = getenv("USERPROFILE");
  char plans[1024];
  snprintf(plans, sizeof(plans), "%s/.cache/ekvido-vkfft-tests", home ? home : ".");
  for(char *c = plans + 1; *c; c++)
    if(*c == '/' || *c == '\\')
    {
      const char sep = *c;
      *c = 0;
      _mkdir(plans); /* parents may exist already */
      *c = sep;
    }
  _mkdir(plans);
#else
  const char *home = getenv("HOME");
  char plans[1024];
  snprintf(plans, sizeof(plans), "%s/.cache/ekvido-vkfft-tests", home ? home : ".");
  char mk[1100];
  snprintf(mk, sizeof(mk), "mkdir -p '%s'", plans);
  if(system(mk)) return 1;
#endif
  E = ekv_cl_new(ctx, dev, q, src, plans);
  free(src);
  printf("device: %s\n", name);
  CHECK(E != NULL, "ekvido.cl builds and every kernel exists");
  if(!E) return 1;
  if(argc == 5 && !strcmp(argv[2], "bench"))
  {
    bench(atoi(argv[3]), atoi(argv[4]));
  }
  else
  {
    test_fft(90, 120);
    test_fft(5832, 8000);
    test_fft(8000, 5832); /* portrait: axis 0 short, axis 1 long */
    test_fft_plan_cache();
    test_split_energy_mask();
    test_clip_level_and_hash();
    test_blur();
    test_warmth();
    test_grid_conv();
    test_basis_and_mac();
    test_outputs();
    test_kazuki_pipeline();
    test_kazuki_cache();
    test_mist_pipeline_and_cache();
    test_tiny_and_dark();
    CHECK(!E->fft_temp && !E->fft_temp_bytes, "no FFT scratch held between pipeline runs (plans hold kernels only)");
  }
  ekv_cl_free(E);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);
  printf("%d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
