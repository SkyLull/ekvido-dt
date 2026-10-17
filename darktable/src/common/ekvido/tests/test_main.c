/* Standalone tests for the ekvido core: unit checks + Python-oracle fixtures.
   Usage: ekvido_core_tests <fixture dir> */
#include "../blend.h"
#include "../color.h"
#include "../convolve.h"
#include "../energy.h"
#include "../fft.h"
#include "../kazuki.h"
#include "../kernel_view.h"
#include "../mist.h"
#include "../psf.h"
#include "../roi.h"
#include "../wblur.h"
#include "ekv.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int g_fail = 0, g_checks = 0;
static const char *g_dir = ".";

#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    g_checks++;                                                             \
    if(!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                  fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); }    \
  } while(0)

/* Max |got - want| <= rel * max|want|; non-finite values must match class. */
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

static double kv_float(const char *line, const char *key, double dflt)
{
  char pat[64];
  snprintf(pat, sizeof(pat), " %s=", key);
  const char *p = strstr(line, pat);
  return p ? strtod(p + strlen(pat), NULL) : dflt;
}

static int load(const char *name, const char *suffix, ekv_t *a)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s.%s.ekv", g_dir, name, suffix);
  return ekv_read(path, a);
}

static void test_fft_size(void)
{
  const int in[] = { 1, 7, 11, 97, 121, 0 }, want[] = { 1, 8, 12, 100, 125, 1 };
  for(int i = 0; i < 6; i++)
    CHECK(ekvido_fft_size(in[i]) == want[i], "fft_size(%d) = %d, want %d", in[i], ekvido_fft_size(in[i]), want[i]);
}

static void test_fft_round_trip(void)
{
  enum { H = 12, W = 10 };
  float x[H * W], y[H * W], spec[H * (W / 2 + 1) * 2];
  for(int i = 0; i < H * W; i++) x[i] = (float)((i * 37) % 11) - 3.f;
  ekvido_rfft2(x, spec, H, W, 2);
  ekvido_irfft2(spec, y, H, W, 2);
  CHECK(close_enough("fft round trip", y, x, H * W, 1e-5), "rfft2/irfft2 round trip");
  /* DC term = sum of inputs */
  double s = 0;
  for(int i = 0; i < H * W; i++) s += x[i];
  CHECK(fabs(spec[0] - s) < 1e-3 && fabs(spec[1]) < 1e-3, "DC term %g + %gi, want %g", spec[0], spec[1], s);
}

static void spec_from_line(const char *line, ekvido_kernel_spec_t *s)
{
  s->halo_radius = kv_float(line, "halo_radius", 0.004);
  s->halo_beta = kv_float(line, "halo_beta", 1.5);
  s->halo_stretch_x = kv_float(line, "halo_stretch_x", 1.0);
  s->halo_stretch_y = kv_float(line, "halo_stretch_y", 1.0);
  s->veil_radius = kv_float(line, "veil_radius", 0.15);
  s->veil_beta = kv_float(line, "veil_beta", 1.0);
  s->arms = (int)kv_float(line, "arms", 0);
  s->arm_angle = kv_float(line, "arm_angle", 0.0);
  s->arm_radius = kv_float(line, "arm_radius", 0.001);
  s->arm_beta = kv_float(line, "arm_beta", 1.5);
  s->arm_sharpness = kv_float(line, "arm_sharpness", 0.4);
  s->arm_stretch_x = kv_float(line, "arm_stretch_x", 1.0);
  s->arm_stretch_y = kv_float(line, "arm_stretch_y", 1.0);
  s->w_halo = kv_float(line, "w_halo", 1.0);
  s->w_veil = kv_float(line, "w_veil", 0.0);
  s->w_arms = kv_float(line, "w_arms", 0.0);
}

static void case_psf(const char *line, const char *name)
{
  ekv_t want;
  if(load(name, "kernel", &want)) { CHECK(0, "psf %s: missing fixture", name); return; }
  ekvido_kernel_spec_t s;
  spec_from_line(line, &s);
  const float diag = (float)kv_float(line, "diag", 400);
  int r = -1;
  float *k = ekvido_build_kernel(&s, diag, &r);
  CHECK(k && 2 * r + 1 == want.dims[0], "psf %s: size %d, want %d", name, 2 * r + 1, want.dims[0]);
  CHECK(r == ekvido_kernel_half_size(&s, diag), "psf %s: radius disagrees with half_size", name);
  if(k && 2 * r + 1 == want.dims[0])
    CHECK(close_enough(name, k, want.data, want.n, 1e-5), "psf %s values", name);
  free(k);
  ekv_free(&want);
}

/* Review Focus 3: extreme sliders still give a finite, normalised, capped kernel. */
static void test_psf_extremes(void)
{
  const float diag = 300.f;
  const ekvido_kernel_spec_t specs[] = {
    { 0.05f, 0.6f, 5.f, 0.25f, 0.25f, 0.6f, 8, 179.f, 0.02f, 0.6f, 0.2f, 5.f, 0.25f, 1.f, 1.f, 1.f },
    { 0.0005f, 4.f, 0.25f, 5.f, 0.02f, 3.f, 1, 0.f, 0.0001f, 4.f, 2.f, 0.25f, 5.f, 0.f, 0.f, 1.f },
    { 0.004f, 1.5f, 1.f, 1.f, 0.15f, 1.f, 8, 45.f, 0.001f, 1.3f, 0.2f, 1.f, 1.f, 0.f, 0.f, 1.f },
  };
  for(int t = 0; t < 3; t++)
  {
    int r = 0;
    float *k = ekvido_build_kernel(&specs[t], diag, &r);
    CHECK(k != NULL, "extreme %d: NULL", t);
    if(!k) continue;
    const size_t n = (size_t)(2 * r + 1) * (2 * r + 1);
    double sum = 0;
    int finite = 1;
    for(size_t i = 0; i < n; i++) { finite &= isfinite(k[i]) && k[i] >= 0.f; sum += k[i]; }
    CHECK(finite, "extreme %d: non-finite or negative values", t);
    CHECK(fabs(sum - 1.0) < 1e-4, "extreme %d: sum %g", t, sum);
    CHECK(r <= ekvido_kernel_half_cap(diag), "extreme %d: radius %d over cap", t, r);
    free(k);
  }
}

/* Phase 1 Review Focus 2: the kernel and its bands do not depend on the thread count. */
static void test_kernel_threads(void)
{
#ifdef _OPENMP
  const ekvido_kernel_spec_t s = { 0.006f, 1.3f, 1.85f, 1.f, 0.1f, 1.2f, 4, 17.f, 0.001f, 0.96f, 0.32f, 2.93f, 0.58f,
                                   1.f, 0.3f, 0.2f };
  const int saved = omp_get_max_threads();
  int r1 = 0, r4 = 0;
  omp_set_num_threads(1);
  float *k1 = ekvido_build_kernel(&s, 900.f, &r1);
  omp_set_num_threads(4);
  float *k4 = ekvido_build_kernel(&s, 900.f, &r4);
  const size_t n = (size_t)(2 * r1 + 1) * (2 * r1 + 1);
  CHECK(k1 && k4 && r1 == r4 && !memcmp(k1, k4, sizeof(float) * n), "kernel depends on the thread count");
  float *b1 = malloc(sizeof(float) * n * 2), *b4 = malloc(sizeof(float) * n * 2);
  omp_set_num_threads(1);
  ekvido_split_bands(k1, r1, 5.4, b1, b1 + n);
  omp_set_num_threads(4);
  ekvido_split_bands(k1, r1, 5.4, b4, b4 + n);
  CHECK(!memcmp(b1, b4, sizeof(float) * n * 2), "bands depend on the thread count");
  omp_set_num_threads(saved);
  free(k1); free(k4); free(b1); free(b4);
#endif
}

static void case_conv(const char *line, const char *name)
{
  (void)line;
  ekv_t src, ker, want;
  if(load(name, "src", &src) || load(name, "kernel", &ker) || load(name, "out", &want))
  { CHECK(0, "conv %s: missing fixture", name); return; }
  const int C = src.dims[0], H = src.dims[1], W = src.dims[2], r = (ker.dims[0] - 1) / 2;
  const float *planes[8];
  float *outs[8];
  float *got = malloc(sizeof(float) * src.n);
  for(int c = 0; c < C; c++) { planes[c] = src.data + (size_t)c * H * W; outs[c] = got + (size_t)c * H * W; }
  for(int pad = r; pad <= r + 7; pad += 7) /* exact fit and a larger pad give the same result */
  {
    ekvido_conv_t *cv = ekvido_conv_new(planes, C, H, W, pad, 2);
    CHECK(cv && ekvido_conv_apply(cv, ker.data, r, outs) == 0, "conv %s: apply failed (pad %d)", name, pad);
    CHECK(close_enough(name, got, want.data, want.n, 5e-5), "conv %s values (pad %d)", name, pad);
    CHECK(ekvido_conv_apply(cv, ker.data, pad + 1, outs) == -1, "conv %s: oversize kernel accepted", name);
    ekvido_conv_free(cv);
  }
  free(got);
  ekv_free(&src); ekv_free(&ker); ekv_free(&want);
}

static void test_coverage(void)
{
  /* 3x3 box kernel: interior coverage 1, corners 4/9, edges 6/9 */
  const float k[9] = { 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9 };
  float cov[6 * 5];
  CHECK(ekvido_conv_coverage(6, 5, 1, k, 1, cov, 1) == 0, "coverage failed");
  CHECK(fabsf(cov[2 * 5 + 2] - 1.f) < 1e-5f, "interior coverage %g", cov[2 * 5 + 2]);
  CHECK(fabsf(cov[0] - 4.f / 9) < 1e-5f, "corner coverage %g", cov[0]);
  CHECK(fabsf(cov[2 * 5] - 6.f / 9) < 1e-5f, "edge coverage %g", cov[2 * 5]);
}

/* Review Focus 1: the margin request for zoomed views. */
static void test_expand_rect(void)
{
  ekvido_rect_t r = ekvido_expand_rect((ekvido_rect_t){ 100, 50, 40, 30 }, 20, 1000, 800);
  CHECK(r.x == 80 && r.y == 30 && r.w == 80 && r.h == 70, "interior expand %d,%d %dx%d", r.x, r.y, r.w, r.h);
  r = ekvido_expand_rect((ekvido_rect_t){ 5, 0, 40, 30 }, 20, 60, 40);
  CHECK(r.x == 0 && r.y == 0 && r.w == 60 && r.h == 40, "clamped expand %d,%d %dx%d", r.x, r.y, r.w, r.h);
  r = ekvido_expand_rect((ekvido_rect_t){ 0, 0, 64, 48 }, 0, 63, 47); /* rounding: out larger than full */
  CHECK(r.w == 64 && r.h == 48, "never smaller than out: %dx%d", r.w, r.h);
}

/* (3,H,W) planar RGB -> H*W*4 RGBA interleaved, alpha 0.5 */
static float *to_rgba(const ekv_t *img)
{
  const int H = img->dims[1], W = img->dims[2];
  const size_t np = (size_t)H * W;
  float *rgba = malloc(sizeof(float) * np * 4);
  for(size_t i = 0; i < np; i++)
  {
    for(int c = 0; c < 3; c++) rgba[i * 4 + c] = img->data[c * np + i];
    rgba[i * 4 + 3] = 0.5f;
  }
  return rgba;
}

/* RGBA rect -> planar RGB (for comparing with fixtures) */
static float *rgb_planes(const float *rgba, int w, int h)
{
  const size_t np = (size_t)w * h;
  float *p = malloc(sizeof(float) * np * 3);
  for(size_t i = 0; i < np; i++)
    for(int c = 0; c < 3; c++) p[c * np + i] = rgba[i * 4 + c];
  return p;
}

static void case_mist(const char *line, const char *name)
{
  ekv_t in, want, glow;
  if(load(name, "in", &in) || load(name, "out", &want) || load(name, "glow", &glow))
  { CHECK(0, "mist %s: missing fixture", name); return; }
  const int H = in.dims[1], W = in.dims[2];
  ekvido_mist_params_t p;
  spec_from_line(line, &p.spec);
  p.strength = (float)kv_float(line, "strength", 0.15);
  const float diag = (float)kv_float(line, "diag", 80);
  float *rgba = to_rgba(&in), *out = malloc(sizeof(float) * (size_t)H * W * 4);
  const ekvido_rect_t full = { 0, 0, W, H };
  ekvido_mist_cache_t *cache = ekvido_mist_cache_new();
  for(int variant = 0; variant < 4; variant++) /* pad fit/cap x no cache/cache */
  {
    ekvido_mist_cache_t *cc = variant & 2 ? cache : NULL;
    CHECK(ekvido_mist_process(rgba, W, H, out, full, &p, diag, variant & 1, 0, cc, 2) == 0, "mist %s: failed", name);
    float *rgb = rgb_planes(out, W, H);
    CHECK(close_enough(name, rgb, want.data, want.n, 5e-5), "mist %s final (variant %d)", name, variant);
    free(rgb);
    int alpha_ok = 1;
    for(size_t i = 0; i < (size_t)H * W; i++) alpha_ok &= out[i * 4 + 3] == 0.5f;
    CHECK(alpha_ok, "mist %s: alpha not preserved", name);
  }
  CHECK(ekvido_mist_process(rgba, W, H, out, full, &p, diag, 1, 1, cache, 2) == 0, "mist %s glow failed", name);
  CHECK(ekvido_mist_cache_glow_hits(cache) >= 1, "mist %s: glow cache never hit", name);
  float *rgb = rgb_planes(out, W, H);
  CHECK(close_enough("glow", rgb, glow.data, glow.n, 5e-5), "mist %s glow-only", name);
  free(rgb);

  /* Review Focus 1: a sub-buffer with a full kernel margin around the output gives the same pixels. */
  const int r = ekvido_kernel_half_size(&p.spec, diag);
  const ekvido_rect_t want_rect = { 20, 18, 16, 10 };
  const ekvido_rect_t in_rect = ekvido_expand_rect(want_rect, r, W, H);
  float *sub = malloc(sizeof(float) * (size_t)in_rect.w * in_rect.h * 4);
  for(int y = 0; y < in_rect.h; y++)
    memcpy(sub + (size_t)y * in_rect.w * 4, rgba + ((size_t)(in_rect.y + y) * W + in_rect.x) * 4,
           sizeof(float) * (size_t)in_rect.w * 4);
  const ekvido_rect_t inner = { want_rect.x - in_rect.x, want_rect.y - in_rect.y, want_rect.w, want_rect.h };
  float *sub_out = malloc(sizeof(float) * (size_t)want_rect.w * want_rect.h * 4);
  CHECK(ekvido_mist_process(sub, in_rect.w, in_rect.h, sub_out, inner, &p, diag, 0, 0, NULL, 2) == 0, "margin run failed");
  ekvido_mist_process(rgba, W, H, out, full, &p, diag, 0, 0, NULL, 2);
  float *crop = malloc(sizeof(float) * (size_t)want_rect.w * want_rect.h * 4);
  for(int y = 0; y < want_rect.h; y++)
    memcpy(crop + (size_t)y * want_rect.w * 4, out + ((size_t)(want_rect.y + y) * W + want_rect.x) * 4,
           sizeof(float) * (size_t)want_rect.w * 4);
  CHECK(close_enough("margin", sub_out, crop, (size_t)want_rect.w * want_rect.h * 4, 5e-5), "mist %s: margin result differs", name);
  free(sub); free(sub_out); free(crop);

  ekvido_mist_cache_free(cache);
  free(rgba); free(out);
  ekv_free(&in); ekv_free(&want); ekv_free(&glow);
}

/* Review Focus 4 and 5: strength 0 is an exact copy; cache invalidates on shape and input changes. */
static void test_mist_identity_and_cache(void)
{
  enum { W = 40, H = 30 };
  float in[W * H * 4], out[W * H * 4], a[W * H * 4], b[W * H * 4];
  for(int i = 0; i < W * H * 4; i++) in[i] = (float)((i * 13) % 17) * 0.1f;
  in[(15 * W + 20) * 4] = 50.f;
  ekvido_mist_params_t p = { 0.f, { 0.004f, 1.5f, 1.f, 1.f, 0.15f, 1.f, 0, 0.f, 0.001f, 1.3f, 0.3f, 1.f, 1.f, 0.5f, 0.5f, 0.f } };
  const ekvido_rect_t full = { 0, 0, W, H };
  const float diag = 50.f;
  CHECK(ekvido_mist_process(in, W, H, out, full, &p, diag, 0, 0, NULL, 1) == 0, "strength 0 failed");
  CHECK(!memcmp(in, out, sizeof(in)), "strength 0 is not an exact copy");

  ekvido_mist_cache_t *c = ekvido_mist_cache_new();
  p.strength = 0.3f;
  ekvido_mist_process(in, W, H, a, full, &p, diag, 1, 0, c, 1);
  p.strength = 0.6f; /* strength-only change: glow reused */
  ekvido_mist_process(in, W, H, b, full, &p, diag, 1, 0, c, 1);
  CHECK(ekvido_mist_cache_glow_hits(c) == 1, "strength change did not reuse the glow");
  p.spec.halo_radius = 0.02f; /* shape change: must recompute */
  ekvido_mist_process(in, W, H, a, full, &p, diag, 1, 0, c, 1);
  ekvido_mist_process(in, W, H, b, full, &p, diag, 1, 0, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(a)), "shape change served a stale glow");
  in[(5 * W + 5) * 4 + 1] = 40.f; /* input change: must recompute */
  ekvido_mist_process(in, W, H, a, full, &p, diag, 1, 0, c, 1);
  ekvido_mist_process(in, W, H, b, full, &p, diag, 1, 0, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(a)), "input change served a stale glow");
  ekvido_mist_cache_free(c);
}

static void case_kview(const char *line, const char *name)
{
  (void)line;
  ekv_t k, want;
  if(load(name, "kernel", &k) || load(name, "u8", &want)) { CHECK(0, "kview %s: missing fixture", name); return; }
  unsigned char *g = malloc(k.n);
  ekvido_kernel_log_u8(k.data, k.n, g);
  int worst = 0;
  for(size_t i = 0; i < k.n; i++) worst = abs((int)g[i] - (int)want.data[i]) > worst ? abs((int)g[i] - (int)want.data[i]) : worst;
  CHECK(worst <= 1, "kview %s: max u8 difference %d", name, worst);
  free(g);
  ekv_free(&k); ekv_free(&want);
}

static void test_kernel_view(void)
{
  const int W = 101;
  unsigned char *img = malloc((size_t)W * W * 4);
  const ekvido_kernel_spec_t s = { 0.006f, 1.3f, 1.85f, 1.f, 0.1f, 1.2f, 4, 0.f, 0.001f, 0.96f, 0.32f, 2.93f, 0.58f, 1.f, 0.f, 0.2f };
  const float diag = ekvido_kernel_view(&s, W, img);
  const unsigned char *c = img + ((size_t)50 * W + 50) * 4;
  CHECK(diag > 0.f, "kernel view diag %g", diag);
  CHECK(c[0] == 255 && c[1] == 255 && c[2] == 255 && c[3] == 255, "centre pixel %d,%d,%d,%d", c[0], c[1], c[2], c[3]);
  CHECK(img[3] == 255 && img[0] == img[1], "corner must be opaque grey/black");
  free(img);
}

/* ---- kazuki stages (Plan B) ---- */

static void kv_str(const char *line, const char *key, char *buf, size_t n)
{
  char pat[64];
  snprintf(pat, sizeof(pat), " %s=", key);
  const char *p = strstr(line, pat);
  buf[0] = 0;
  if(!p) return;
  p += strlen(pat);
  size_t k = 0;
  while(p[k] && p[k] != ' ' && k + 1 < n) { buf[k] = p[k]; k++; }
  buf[k] = 0;
}

/* `<input>.<suffix>.ekv`, where input= names the shared fixture */
static int load_input(const char *line, const char *suffix, ekv_t *a)
{
  char base[128];
  kv_str(line, "input", base, sizeof(base));
  return load(base, suffix, a);
}

/* (3,H,W) planar -> positive planes (malloc'd 3*H*W) */
static float *positive_planes(const ekv_t *img)
{
  float *p = malloc(sizeof(float) * img->n);
  for(size_t i = 0; i < img->n; i++) p[i] = ekvido_positive(img->data[i]);
  return p;
}

static void case_energy(const char *line, const char *name)
{
  ekv_t in, want;
  if(load_input(line, "in", &in) || load(name, "energy", &want)) { CHECK(0, "energy %s: missing fixture", name); return; }
  const size_t np = (size_t)in.dims[1] * in.dims[2];
  float *src = positive_planes(&in), *got = malloc(sizeof(float) * in.n);
  const float *s3[3] = { src, src + np, src + 2 * np };
  float *e3[3] = { got, got + np, got + 2 * np };
  ekvido_extract_energy(s3, np, (float)kv_float(line, "thr", 0), (float)kv_float(line, "knee", 1), e3);
  CHECK(close_enough(name, got, want.data, want.n, 1e-5), "energy %s values", name);
  free(src); free(got);
  ekv_free(&in); ekv_free(&want);
}

static void case_clip(const char *line, const char *name)
{
  ekv_t in;
  if(load_input(line, "in", &in)) { CHECK(0, "clip %s: missing fixture", name); return; }
  float *rgba = to_rgba(&in);
  const float got = ekvido_estimate_clip_level(rgba, (size_t)in.dims[1] * in.dims[2], 4);
  const double want = kv_float(line, "clip", -1);
  if(isinf(want)) CHECK(isinf(got) && got > 0, "clip %s: got %g, want inf", name, got);
  else CHECK(fabs(got - want) <= 1e-6 * fabs(want), "clip %s: got %.9g, want %.9g", name, got, want);
  free(rgba);
  ekv_free(&in);
}

static void case_blur(const char *line, const char *name)
{
  ekv_t in, want;
  if(load_input(line, "in", &in) || load(name, "out", &want)) { CHECK(0, "blur %s: missing fixture", name); return; }
  const int H = in.dims[0], W = in.dims[1];
  const double sigma = kv_float(line, "sigma", 1);
  float *got = malloc(sizeof(float) * in.n);
  CHECK(ekvido_gaussian_blur(in.data, got, H, W, sigma) == 0, "blur %s failed", name);
  CHECK(close_enough(name, got, want.data, want.n, 1e-5), "blur %s values", name);
  memcpy(got, in.data, sizeof(float) * in.n); /* in place gives the same */
  ekvido_gaussian_blur(got, got, H, W, sigma);
  CHECK(close_enough(name, got, want.data, want.n, 1e-5), "blur %s in place", name);
  free(got);
  ekv_free(&in); ekv_free(&want);
}

static void case_wblur(const char *line, const char *name)
{
  ekv_t x, w, want, wden;
  if(load_input(line, "x", &x) || load_input(line, "w", &w) || load(name, "out", &want) || load(name, "den", &wden))
  { CHECK(0, "wblur %s: missing fixture", name); return; }
  const int H = x.dims[1], W = x.dims[2];
  const size_t np = (size_t)H * W;
  float *got = malloc(sizeof(float) * x.n), *den = malloc(sizeof(float) * np);
  const float *xs[2] = { x.data, x.data + np };
  float *outs[2] = { got, got + np };
  CHECK(ekvido_weighted_blur(xs, 2, w.data, H, W, kv_float(line, "sigma", 1), outs, den) == 0, "wblur %s failed", name);
  CHECK(close_enough(name, got, want.data, want.n, 1e-5), "wblur %s values", name);
  CHECK(close_enough("den", den, wden.data, wden.n, 1e-5), "wblur %s den", name);
  free(got); free(den);
  ekv_free(&x); ekv_free(&w); ekv_free(&want); ekv_free(&wden);
}

static void case_blend(const char *line, const char *name)
{
  ekv_t img, glow, energy, want;
  if(load_input(line, "in", &img) || load("blend_input", "glow", &glow) || load("blend_input", "energy", &energy)
     || load(name, "out", &want))
  { CHECK(0, "blend %s: missing fixture", name); return; }
  const ekvido_blend_mode_t mode = (ekvido_blend_mode_t)(int)kv_float(line, "mode", 0);
  const float strength = (float)kv_float(line, "strength", 0.5);
  const float white = (float)exp2(kv_float(line, "white_ev", 0));
  const size_t np = (size_t)img.dims[1] * img.dims[2];
  float *got = malloc(sizeof(float) * img.n);
  for(size_t i = 0; i < np; i++) /* planar fixtures -> one RGB pixel at a time */
  {
    const float s[3] = { img.data[i], img.data[np + i], img.data[2 * np + i] };
    const float g[3] = { glow.data[i], glow.data[np + i], glow.data[2 * np + i] };
    const float e[3] = { energy.data[i], energy.data[np + i], energy.data[2 * np + i] };
    float o[3];
    ekvido_blend_px(mode, s, g, e, strength, white, o);
    for(int c = 0; c < 3; c++) got[c * np + i] = o[c];
  }
  CHECK(close_enough(name, got, want.data, want.n, 1e-5), "blend %s values", name);
  free(got);
  ekv_free(&img); ekv_free(&glow); ekv_free(&energy); ekv_free(&want);
}

static void test_blend_formulas(void)
{
  CHECK(fabsf(ekvido_csp_blend(EKVIDO_BLEND_SCREEN, 0.5f, 0.5f, 1.f) - 0.75f) < 1e-6f, "screen");
  CHECK(fabsf(ekvido_csp_blend(EKVIDO_BLEND_OVERLAY, 0.25f, 0.5f, 1.f) - 0.25f) < 1e-6f, "overlay low");
  CHECK(fabsf(ekvido_csp_blend(EKVIDO_BLEND_OVERLAY, 0.75f, 1.f, 0.5f) - 0.875f) < 1e-6f, "overlay high");
  CHECK(fabsf(ekvido_csp_blend(EKVIDO_BLEND_COLOR_DODGE, 0.2f, 0.5f, 0.5f) - 0.3f) < 1e-6f, "colour dodge");
  CHECK(fabsf(ekvido_csp_blend(EKVIDO_BLEND_GLOW_DODGE, 0.2f, 0.5f, 0.5f) - 0.2f / 0.75f) < 1e-6f, "glow dodge");
  /* overlay (glow only), hand cases as in Python test_overlay_glow_hand_cases */
  const float lo_c[3] = { 0.4f, 0.2f, 0.f }, lo_w[3] = { 0.35f, 0.25f, 0.15f };
  const float hi_c[3] = { 0.5f, 0.5f, 0.25f }, hi_w[3] = { 0.8125f, 0.8125f, 0.75f };
  for(int c = 0; c < 3; c++)
  {
    CHECK(fabsf(ekvido_overlay_glow(0.25f, lo_c[c], 0.4f, 1.f) - lo_w[c]) < 1e-6f, "overlay glow low [%d]", c);
    CHECK(fabsf(ekvido_overlay_glow(0.75f, hi_c[c], 0.5f, 0.5f) - hi_w[c]) < 1e-6f, "overlay glow high [%d]", c);
  }
  CHECK(ekvido_overlay_glow(0.3f, 0.6f, 1.f, 0.7f) == ekvido_csp_blend(EKVIDO_BLEND_OVERLAY, 0.3f, 0.6f, 0.7f), "m = 1 is overlay");
  CHECK(ekvido_overlay_glow(0.3f, 1e-9f, 1e-9f, 1.f) == 0.3f && ekvido_overlay_glow(0.3f, 0.f, 0.f, 1.f) == 0.3f, "no glow");
  CHECK(isfinite(ekvido_overlay_glow(0.3f, 2e-6f, 2e-6f, 1.f)) && fabsf(ekvido_overlay_glow(0.3f, 2e-6f, 2e-6f, 1.f) - 0.3f) < 1e-5f, "tiny glow");
  {
    const float img[3] = { 0.02f, -0.1f, 7.f }, zero[3] = { 0.f, 0.f, 0.f };
    float o[3];
    ekvido_blend_px(EKVIDO_BLEND_OVERLAY_GLOW, img, zero, zero, 0.8f, 1.f, o);
    CHECK(!memcmp(o, img, sizeof(o)), "glow-free pixel changed by overlay (glow only)");
  }
  CHECK(ekvido_dodge(0.f, 0.9f) == 0.f && ekvido_dodge(0.3f, 1.f) == 1.f && ekvido_dodge(0.9f, 0.2f) == 1.f, "dodge cases");
  for(int m = EKVIDO_BLEND_SCREEN; m <= EKVIDO_BLEND_OVERLAY_GLOW; m++)
  {
    /* above white passes through; strength 0 is the input; NaN stays NaN */
    CHECK(fabsf(ekvido_blend((ekvido_blend_mode_t)m, 40.f, 9.f, 0.f, 0.6f, 1.f) - 40.f) < 1e-4f, "mode %d above white", m);
    CHECK(ekvido_blend((ekvido_blend_mode_t)m, 0.3f, 0.5f, 0.f, 0.f, 1.f) == 0.3f, "mode %d strength 0", m);
    CHECK(isnan(ekvido_blend((ekvido_blend_mode_t)m, NAN, 0.5f, 0.f, 0.6f, 1.f)), "mode %d NaN", m);
    CHECK(isinf(ekvido_blend((ekvido_blend_mode_t)m, INFINITY, 0.5f, 0.f, 0.6f, 1.f)), "mode %d inf", m);
  }
}

static void case_warmth(const char *line, const char *name)
{
  ekv_t in, want_t, want_c;
  if(load_input(line, "in", &in) || load(name, "t", &want_t) || load(name, "clipped", &want_c))
  { CHECK(0, "warmth %s: missing fixture", name); return; }
  const int H = in.dims[1], W = in.dims[2];
  const size_t np = (size_t)H * W;
  float *src = positive_planes(&in), *t = malloc(sizeof(float) * np);
  unsigned char *cl = malloc(np);
  const float *s3[3] = { src, src + np, src + 2 * np };
  const double sigma = kv_float(line, "hue_radius", 0.01) * kv_float(line, "diag", 100);
  CHECK(ekvido_warmth_map(s3, H, W, (float)kv_float(line, "clip", INFINITY), sigma, (float)kv_float(line, "y_floor", 1),
                          (float)kv_float(line, "neutral_point", 0), (float)kv_float(line, "warmth_softness", 0.05),
                          (float)kv_float(line, "neutral_bias", 0.8), t, cl) == 0, "warmth %s failed", name);
  CHECK(close_enough(name, t, want_t.data, want_t.n, 1e-5), "warmth %s t", name);
  int same = 1;
  for(size_t i = 0; i < np; i++) same &= (float)cl[i] == want_c.data[i];
  CHECK(same, "warmth %s: clipped mask differs", name);
  free(src); free(t); free(cl);
  ekv_free(&in); ekv_free(&want_t); ekv_free(&want_c);
}

static void case_bands(const char *line, const char *name)
{
  ekv_t k, wi, wo;
  if(load_input(line, "kernel", &k) || load(name, "inner", &wi) || load(name, "outer", &wo))
  { CHECK(0, "bands %s: missing fixture", name); return; }
  const int r = (k.dims[0] - 1) / 2;
  float *inner = malloc(sizeof(float) * k.n), *outer = malloc(sizeof(float) * k.n);
  ekvido_split_bands(k.data, r, kv_float(line, "split", 0), inner, outer);
  CHECK(close_enough("inner", inner, wi.data, wi.n, 1e-5), "bands %s inner", name);
  CHECK(close_enough("outer", outer, wo.data, wo.n, 1e-5), "bands %s outer", name);
  free(inner); free(outer);
  ekv_free(&k); ekv_free(&wi); ekv_free(&wo);
}

static void case_hsv(const char *line, const char *name)
{
  float rgb[3];
  ekvido_hsv_to_rgb((float)kv_float(line, "hue", 0), (float)kv_float(line, "sat", 0), rgb);
  const double want[3] = { kv_float(line, "r", -1), kv_float(line, "g", -1), kv_float(line, "b", -1) };
  for(int c = 0; c < 3; c++) CHECK(fabs(rgb[c] - want[c]) < 1e-6, "hsv %s[%d]: %g vs %g", name, c, rgb[c], want[c]);
}

static void test_warmth_colour(void)
{
  float c[3];
  ekvido_warmth_colour(0.f, c);
  CHECK(fabsf(c[0] - ekvido_srgb_decode(40.f / 255.f)) < 1e-7f && c[2] == 1.f, "cool end %g %g %g", c[0], c[1], c[2]);
  ekvido_warmth_colour(0.5f, c);
  CHECK(c[0] == c[1] && c[1] == c[2] && fabsf(c[0] - ekvido_srgb_decode(128.f / 255.f)) < 1e-7f, "neutral grey");
  ekvido_warmth_colour(1.f, c);
  CHECK(c[0] == 1.f && fabsf(c[2] - ekvido_srgb_decode(30.f / 255.f)) < 1e-7f, "warm end");
}

static void kazuki_params_from_line(const char *line, ekvido_kazuki_params_t *p)
{
  spec_from_line(line, &p->spec);
  p->strength = (float)kv_float(line, "strength", 0.5);
  p->blend = (ekvido_blend_mode_t)(int)kv_float(line, "mode", 0);
  p->blend_white_ev = (float)kv_float(line, "blend_white_ev", 0);
  p->threshold_ev = (float)kv_float(line, "threshold_ev", 0);
  p->knee_ev = (float)kv_float(line, "knee_ev", 1);
  p->hue_radius = (float)kv_float(line, "hue_radius", 0.01);
  p->neutral_point = (float)kv_float(line, "neutral_point", 0);
  p->warmth_softness = (float)kv_float(line, "warmth_softness", 0.05);
  p->neutral_bias = (float)kv_float(line, "neutral_bias", 0.8);
  p->push_warm = (float)kv_float(line, "push_warm", 0.7);
  p->push_cool = (float)kv_float(line, "push_cool", 0.7);
  p->band_split = (float)kv_float(line, "band_split", 0.006);
  ekvido_hsv_to_rgb((float)kv_float(line, "warm_core_hue", 0), (float)kv_float(line, "warm_core_sat", 0), p->warm_core);
  ekvido_hsv_to_rgb((float)kv_float(line, "warm_edge_hue", 0), (float)kv_float(line, "warm_edge_sat", 0), p->warm_edge);
  ekvido_hsv_to_rgb((float)kv_float(line, "cool_core_hue", 0), (float)kv_float(line, "cool_core_sat", 0), p->cool_core);
  ekvido_hsv_to_rgb((float)kv_float(line, "cool_edge_hue", 0), (float)kv_float(line, "cool_edge_sat", 0), p->cool_edge);
}

static void case_kazuki(const char *line, const char *name)
{
  ekv_t in, want, wglow, wenergy;
  if(load_input(line, "in", &in) || load(name, "out", &want) || load(name, "glow", &wglow) || load(name, "energy", &wenergy))
  { CHECK(0, "kazuki %s: missing fixture", name); return; }
  const int H = in.dims[1], W = in.dims[2];
  /* source mask (spec §5.7): mask=<base> loads <base>.mask.ekv (H,W); invert=0|1 */
  char mask_base[128];
  kv_str(line, "mask", mask_base, sizeof(mask_base));
  ekv_t mk = { 0 };
  if(mask_base[0] && load(mask_base, "mask", &mk)) { CHECK(0, "kazuki %s: missing mask", name); return; }
  const float *mask = mask_base[0] ? mk.data : NULL;
  const int invert = (int)kv_float(line, "invert", 0);
  ekvido_kazuki_params_t p;
  kazuki_params_from_line(line, &p);
  const float diag = (float)kv_float(line, "diag", 100), clip = (float)kv_float(line, "clip", INFINITY);
  float *rgba = to_rgba(&in), *out = malloc(sizeof(float) * (size_t)H * W * 4);
  const ekvido_rect_t full = { 0, 0, W, H };
  ekvido_kazuki_cache_t *cache = ekvido_kazuki_cache_new();
  for(int variant = 0; variant < 4; variant++) /* pad fit/cap x no cache/cache */
  {
    ekvido_kazuki_cache_t *cc = variant & 2 ? cache : NULL;
    CHECK(ekvido_kazuki_process_masked(rgba, W, H, out, full, &p, diag, clip, mask, invert, variant & 1, EKVIDO_VIEW_FINAL, cc, 2) == 0,
          "kazuki %s: failed", name);
    float *rgb = rgb_planes(out, W, H);
    CHECK(close_enough(name, rgb, want.data, want.n, 5e-5), "kazuki %s final (variant %d)", name, variant);
    free(rgb);
    int alpha_ok = 1;
    for(size_t i = 0; i < (size_t)H * W; i++) alpha_ok &= out[i * 4 + 3] == 0.5f;
    CHECK(alpha_ok, "kazuki %s: alpha not preserved", name);
  }
  const ekvido_kazuki_view_t views[2] = { EKVIDO_VIEW_GLOW, EKVIDO_VIEW_ENERGY };
  const ekv_t *wants[2] = { &wglow, &wenergy };
  for(int v = 0; v < 2; v++)
  {
    CHECK(ekvido_kazuki_process_masked(rgba, W, H, out, full, &p, diag, clip, mask, invert, 1, views[v], cache, 2) == 0, "kazuki %s view", name);
    float *rgb = rgb_planes(out, W, H);
    CHECK(close_enough(v ? "energy" : "glow", rgb, wants[v]->data, wants[v]->n, 5e-5), "kazuki %s view %d", name, views[v]);
    free(rgb);
  }

  /* Review Focus 1: a sub-buffer with the module's margin gives the same pixels (direct blur only:
     the downsampled blur's blocks start at the buffer origin, so they shift with the crop). */
  if((double)p.hue_radius * diag <= 8.0)
  {
    const int m = ekvido_kazuki_margin(&p, diag);
    const ekvido_rect_t want_rect = { 20, 18, 16, 10 };
    const ekvido_rect_t in_rect = ekvido_expand_rect(want_rect, m, W, H);
    float *sub = malloc(sizeof(float) * (size_t)in_rect.w * in_rect.h * 4);
    for(int y = 0; y < in_rect.h; y++)
      memcpy(sub + (size_t)y * in_rect.w * 4, rgba + ((size_t)(in_rect.y + y) * W + in_rect.x) * 4,
             sizeof(float) * (size_t)in_rect.w * 4);
    float *sub_mask = mask ? malloc(sizeof(float) * (size_t)in_rect.w * in_rect.h) : NULL;
    for(int y = 0; mask && y < in_rect.h; y++)
      memcpy(sub_mask + (size_t)y * in_rect.w, mask + (size_t)(in_rect.y + y) * W + in_rect.x,
             sizeof(float) * (size_t)in_rect.w);
    const ekvido_rect_t inner = { want_rect.x - in_rect.x, want_rect.y - in_rect.y, want_rect.w, want_rect.h };
    float *sub_out = malloc(sizeof(float) * (size_t)want_rect.w * want_rect.h * 4);
    CHECK(ekvido_kazuki_process_masked(sub, in_rect.w, in_rect.h, sub_out, inner, &p, diag, clip, sub_mask, invert, 0,
                                       EKVIDO_VIEW_FINAL, NULL, 2) == 0,
          "kazuki %s margin run failed", name);
    ekvido_kazuki_process_masked(rgba, W, H, out, full, &p, diag, clip, mask, invert, 0, EKVIDO_VIEW_FINAL, NULL, 2);
    float *crop = malloc(sizeof(float) * (size_t)want_rect.w * want_rect.h * 4);
    for(int y = 0; y < want_rect.h; y++)
      memcpy(crop + (size_t)y * want_rect.w * 4, out + ((size_t)(want_rect.y + y) * W + want_rect.x) * 4,
             sizeof(float) * (size_t)want_rect.w * 4);
    CHECK(close_enough("margin", sub_out, crop, (size_t)want_rect.w * want_rect.h * 4, 5e-5), "kazuki %s: margin result differs", name);
    free(sub); free(sub_out); free(crop); free(sub_mask);
  }
  ekvido_kazuki_cache_free(cache);
  free(rgba); free(out);
  ekv_free(&in); ekv_free(&want); ekv_free(&wglow); ekv_free(&wenergy);
  if(mask) ekv_free(&mk);
}

static void case_maskenergy(const char *line, const char *name)
{
  ekv_t in, mk, want;
  char mask_base[128];
  kv_str(line, "mask", mask_base, sizeof(mask_base));
  if(load_input(line, "in", &in) || load(mask_base, "mask", &mk) || load(name, "energy", &want))
  { CHECK(0, "maskenergy %s: missing fixture", name); return; }
  const size_t np = (size_t)in.dims[1] * in.dims[2];
  float *src = positive_planes(&in), *got = malloc(sizeof(float) * in.n);
  const float *s3[3] = { src, src + np, src + 2 * np };
  float *e3[3] = { got, got + np, got + 2 * np };
  ekvido_extract_energy(s3, np, (float)kv_float(line, "thr", 0), (float)kv_float(line, "knee", 1), e3);
  ekvido_mask_energy(e3, mk.data, np, (int)kv_float(line, "invert", 0));
  CHECK(close_enough(name, got, want.data, want.n, 1e-5), "maskenergy %s values", name);
  free(src); free(got);
  ekv_free(&in); ekv_free(&mk); ekv_free(&want);
}

/* Review Focus 4, 5 and the preview modes, on a small synthetic frame. */
static void test_kazuki_cache_views_identity(void)
{
  enum { W = 48, H = 36 };
  const size_t np = (size_t)W * H;
  float *in = malloc(sizeof(float) * np * 4), *a = malloc(sizeof(float) * np * 4), *b = malloc(sizeof(float) * np * 4);
  for(size_t i = 0; i < np; i++)
  {
    for(int c = 0; c < 3; c++) in[i * 4 + c] = 0.05f + 0.01f * (float)((i * 7 + c) % 5);
    in[i * 4 + 3] = 1.f;
  }
  for(int y = 15; y < 18; y++)
    for(int x = 20; x < 23; x++)
      for(int c = 0; c < 3; c++) in[((size_t)y * W + x) * 4 + c] = 20.f; /* clipped plateau */
  in[(5 * W + 40) * 4 + 0] = 6.f; in[(5 * W + 40) * 4 + 1] = 3.f; in[(5 * W + 40) * 4 + 2] = 1.f; /* warm source */
  ekvido_kazuki_params_t p = { 0 };
  p.strength = 0.f; p.blend = EKVIDO_BLEND_ADDITIVE; p.threshold_ev = 0.f; p.knee_ev = 1.f;
  p.hue_radius = 0.02f; p.warmth_softness = 0.05f; p.neutral_bias = 0.8f; p.push_warm = 0.7f; p.push_cool = 0.7f;
  ekvido_hsv_to_rgb(40.f, 0.45f, p.warm_core); ekvido_hsv_to_rgb(22.5f, 0.88f, p.warm_edge);
  ekvido_hsv_to_rgb(193.846154f, 0.65f, p.cool_core); ekvido_hsv_to_rgb(232.173913f, 0.92f, p.cool_edge);
  p.band_split = 0.006f;
  p.spec = (ekvido_kernel_spec_t){ 0.006f, 1.3f, 1.85f, 1.f, 0.1f, 1.2f, 4, 0.f, 0.001f, 0.96f, 0.32f, 2.93f, 0.58f, 1.f, 0.f, 0.2f };
  const ekvido_rect_t full = { 0, 0, W, H };
  const float diag = hypotf(W, H), clip = ekvido_estimate_clip_level(in, np, 4);
  CHECK(clip > 19.f && clip < 20.f, "synthetic clip level %g", clip);

  CHECK(ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 0, EKVIDO_VIEW_FINAL, NULL, 1) == 0, "strength 0 failed");
  CHECK(!memcmp(in, a, sizeof(float) * np * 4), "strength 0 is not an exact copy");

  ekvido_kazuki_cache_t *c = ekvido_kazuki_cache_new();
  int src_hits, kernel_hits, glow_hits;
  p.strength = 0.5f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  /* strength, blend and white: the glow itself is reused, no FFT */
  p.strength = 1.2f; p.blend = EKVIDO_BLEND_GLOW_DODGE; p.blend_white_ev = 1.f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  CHECK(src_hits == 1 && kernel_hits == 0 && glow_hits == 1, "strength/blend change recomputed (src %d kernel %d glow %d)",
        src_hits, kernel_hits, glow_hits);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "cached strength change differs from uncached");
  /* colour and pull: source and kernel spectra reused (3 inverse FFTs) */
  p.push_warm = 0.2f; p.warm_edge[1] = 0.9f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  CHECK(src_hits == 2 && kernel_hits == 1 && glow_hits == 1, "colour change: src %d kernel %d glow %d", src_hits,
        kernel_hits, glow_hits);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "cached colour change differs from uncached");
  /* shape change: kernel spectra rebuilt, source reused */
  p.spec.arms = 6;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  CHECK(src_hits == 3 && kernel_hits == 1 && glow_hits == 1, "shape change: src %d kernel %d glow %d", src_hits,
        kernel_hits, glow_hits);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "shape change served a stale glow");
  /* second shape change (old spectra freed first), then colour-only: a hit on the NEW kernel spectra */
  p.spec.arms = 4; p.spec.arm_angle = 17.f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "second shape change differs from uncached");
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  const int kernel_before = kernel_hits;
  p.push_cool = 0.3f; p.cool_edge[2] = 0.8f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  CHECK(kernel_hits == kernel_before + 1, "colour change after shape changes missed the kernel spectra (%d -> %d)",
        kernel_before, kernel_hits);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "colour change after shape changes differs from uncached");
  /* warmth parameter change: source recomputed */
  p.hue_radius = 0.05f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "warmth change served a stale source");
  /* input change: recomputed */
  in[(30 * W + 5) * 4 + 2] = 9.f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "input change served a stale source");

  /* previews: glow ignores strength and blend; clipped marks the plateau red */
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_GLOW, c, 1);
  p.strength = 0.1f; p.blend = EKVIDO_BLEND_ADDITIVE;
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_GLOW, c, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "glow preview depends on strength/blend");
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_CLIPPED, c, 1);
  const float *pl = a + ((size_t)16 * W + 21) * 4, *bg = a + ((size_t)2 * W + 2) * 4;
  CHECK(pl[0] == 1.f && pl[1] == 0.f && pl[2] == 0.f, "clipped plateau not red");
  CHECK(!memcmp(bg, in + ((size_t)2 * W + 2) * 4, sizeof(float) * 4), "unclipped pixel changed in clipped view");
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_WARMTH, c, 1);
  int in_range = 1;
  for(size_t i = 0; i < np; i++)
    for(int k = 0; k < 3; k++) in_range &= a[i * 4 + k] >= 0.f && a[i * 4 + k] <= 1.f;
  CHECK(in_range, "warmth preview out of 0..1");
  ekvido_kazuki_cache_free(c);

  /* margin: covers the kernel and the warmth blur, never above the cap */
  p.hue_radius = 0.01f;
  const int m = ekvido_kazuki_margin(&p, 1000.f);
  CHECK(m >= ekvido_kernel_half_size(&p.spec, 1000.f) && m <= ekvido_kernel_half_cap(1000.f), "margin %d", m);
  p.spec.w_arms = 0.f; p.spec.halo_radius = 0.0005f; p.spec.halo_beta = 4.f; p.hue_radius = 0.021f;
  const int base = ekvido_kernel_half_size(&p.spec, 1000.f) + 2 * 5; /* sigma 21: blocks of 5 */
  const int md = ekvido_kazuki_margin(&p, 1000.f);
  CHECK(md == base + 63, "margin with downsampled blur %d (kernel+blocks %d)", md, base); /* + ceil(3 sigma) */
  free(in); free(a); free(b);
}


/* Spec §5.7: no mask = ones mask, zero mask = no glow, cache sees mask changes. */
static void test_kazuki_source_mask(void)
{
  enum { W = 48, H = 36 };
  const size_t np = (size_t)W * H;
  float *in = malloc(sizeof(float) * np * 4), *a = malloc(sizeof(float) * np * 4), *b = malloc(sizeof(float) * np * 4);
  float *ones = malloc(sizeof(float) * np), *zeros = calloc(np, sizeof(float)), *half = malloc(sizeof(float) * np);
  for(size_t i = 0; i < np; i++)
  {
    for(int c = 0; c < 3; c++) in[i * 4 + c] = 0.05f + 0.01f * (float)((i * 3 + c) % 7);
    in[i * 4 + 3] = 1.f;
    ones[i] = 1.f;
    half[i] = (i % W) < W / 2 ? 1.f : 0.f; /* left half emits */
  }
  in[(10 * W + 10) * 4] = 9.f; in[(10 * W + 10) * 4 + 1] = 5.f; in[(10 * W + 10) * 4 + 2] = 2.f; /* left source */
  in[(25 * W + 40) * 4] = 2.f; in[(25 * W + 40) * 4 + 1] = 5.f; in[(25 * W + 40) * 4 + 2] = 9.f; /* right source */
  ekvido_kazuki_params_t p = { 0 };
  p.strength = 0.7f; p.blend = EKVIDO_BLEND_ADDITIVE; p.knee_ev = 1.f; p.hue_radius = 0.02f;
  p.warmth_softness = 0.05f; p.neutral_bias = 0.8f; p.push_warm = 0.7f; p.push_cool = 0.7f; p.band_split = 0.006f;
  ekvido_hsv_to_rgb(40.f, 0.45f, p.warm_core); ekvido_hsv_to_rgb(22.5f, 0.88f, p.warm_edge);
  ekvido_hsv_to_rgb(193.846154f, 0.65f, p.cool_core); ekvido_hsv_to_rgb(232.173913f, 0.92f, p.cool_edge);
  p.spec = (ekvido_kernel_spec_t){ 0.006f, 1.3f, 1.85f, 1.f, 0.1f, 1.2f, 4, 0.f, 0.001f, 0.96f, 0.32f, 2.93f, 0.58f, 1.f, 0.f, 0.2f };
  const ekvido_rect_t full = { 0, 0, W, H };
  const float diag = hypotf(W, H);

  ekvido_kazuki_process(in, W, H, a, full, &p, diag, INFINITY, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  ekvido_kazuki_process_masked(in, W, H, b, full, &p, diag, INFINITY, ones, 0, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "a mask of ones changed the output");
  ekvido_kazuki_process_masked(in, W, H, b, full, &p, diag, INFINITY, zeros, 0, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(in, b, sizeof(float) * np * 4), "a mask of zeros still glows");
  ekvido_kazuki_process_masked(in, W, H, b, full, &p, diag, INFINITY, ones, 1, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(in, b, sizeof(float) * np * 4), "an inverted mask of ones still glows");
  ekvido_kazuki_process_masked(in, W, H, b, full, &p, diag, INFINITY, half, 0, 1, EKVIDO_VIEW_ENERGY, NULL, 1);
  CHECK(b[(25 * W + 40) * 4] == 0.f && b[(10 * W + 10) * 4] > 0.f, "energy view does not show the masked energy");

  ekvido_kazuki_cache_t *c = ekvido_kazuki_cache_new();
  int src_hits, kernel_hits, glow_hits;
  ekvido_kazuki_process_masked(in, W, H, a, full, &p, diag, INFINITY, half, 0, 1, EKVIDO_VIEW_FINAL, c, 1);
  half[5] = 0.5f; /* same input, different mask: must recompute */
  ekvido_kazuki_process_masked(in, W, H, a, full, &p, diag, INFINITY, half, 0, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process_masked(in, W, H, b, full, &p, diag, INFINITY, half, 0, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  CHECK(src_hits == 0 && !memcmp(a, b, sizeof(float) * np * 4), "mask change served a stale source (hits %d)", src_hits);
  ekvido_kazuki_process_masked(in, W, H, a, full, &p, diag, INFINITY, half, 1, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process_masked(in, W, H, b, full, &p, diag, INFINITY, half, 1, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  CHECK(src_hits == 0 && !memcmp(a, b, sizeof(float) * np * 4), "invert change served a stale source");
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, INFINITY, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, INFINITY, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "dropping the mask served a stale source");
  p.strength = 0.3f; /* colour-only change with the mask on: still reuses the glow */
  ekvido_kazuki_process_masked(in, W, H, a, full, &p, diag, INFINITY, half, 1, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_process_masked(in, W, H, a, full, &p, diag, INFINITY, half, 1, 1, EKVIDO_VIEW_FINAL, c, 1);
  ekvido_kazuki_cache_hits(c, &src_hits, &kernel_hits, &glow_hits);
  CHECK(src_hits == 1 && glow_hits == 1, "masked repeat did not hit the cache (src %d glow %d)", src_hits, glow_hits);
  ekvido_kazuki_cache_free(c);
  free(in); free(a); free(b); free(ones); free(zeros); free(half);
}

static void fill_random(float *x, size_t n, uint32_t seed)
{
  for(size_t i = 0; i < n; i++)
  {
    seed = seed * 1664525u + 1013904223u;
    x[i] = (float)(seed >> 8) / 16777216.f;
  }
}

/* Phase 1: the grid API gives exactly what ekvido_conv_apply gives. */
static void test_grid_matches_conv(void)
{
  enum { H = 23, W = 31, R = 4, PAD = 7 };
  const int d = 2 * R + 1;
  float *src = malloc(sizeof(float) * H * W), *ker = malloc(sizeof(float) * d * d);
  float *want = malloc(sizeof(float) * H * W), *got = malloc(sizeof(float) * H * W);
  fill_random(src, (size_t)H * W, 11u);
  fill_random(ker, (size_t)d * d, 29u);
  const float *planes[1] = { src };
  float *outs[1] = { want };
  ekvido_conv_t *cv = ekvido_conv_new(planes, 1, H, W, PAD, 1);
  CHECK(cv && ekvido_conv_apply(cv, ker, R, outs) == 0, "grid: conv reference failed");
  const ekvido_grid_t g = ekvido_grid_make(H, W, PAD, 1);
  CHECK(g.fh >= H + PAD && g.fw >= W + PAD && g.h == H && g.w == W && g.pad == PAD, "grid geometry %dx%d", g.fh, g.fw);
  const size_t sf = ekvido_grid_spec_floats(&g);
  CHECK(sf == (size_t)g.fh * (g.fw / 2 + 1) * 2 && ekvido_grid_floats(&g) == (size_t)g.fh * g.fw, "grid sizes");
  float *work = malloc(sizeof(float) * ekvido_grid_floats(&g));
  float *ss = malloc(sizeof(float) * sf), *ks = malloc(sizeof(float) * sf), *prod = malloc(sizeof(float) * sf);
  ekvido_grid_plane_spectrum(&g, src, work, ss);
  CHECK(ekvido_grid_kernel_spectrum(&g, ker, PAD + 1, work, ks) == -1, "grid: oversize kernel accepted");
  CHECK(ekvido_grid_kernel_spectrum(&g, ker, R, work, ks) == 0, "grid: kernel spectrum failed");
  for(size_t q = 0; q < sf; q += 2)
  {
    const float a = ss[q], b = ss[q + 1], e = ks[q], f = ks[q + 1];
    prod[q] = a * e - b * f;
    prod[q + 1] = a * f + b * e;
  }
  ekvido_grid_inverse(&g, prod, work, got);
  CHECK(close_enough("grid", got, want, (size_t)H * W, 5e-5), "grid result differs from ekvido_conv_apply");
  ekvido_conv_free(cv);
  free(src); free(ker); free(want); free(got); free(work); free(ss); free(ks); free(prod);
}

/* Phase 1 Review Focus 4: summed-area coverage = FFT convolution of a ones plane, with an
   asymmetric kernel (catches flipped offsets), a pad above r, and a frame smaller than the kernel. */
static void test_coverage_sat(void)
{
  const int cases[4][4] = { { 6, 5, 1, 1 }, { 40, 27, 6, 6 }, { 40, 27, 6, 11 }, { 3, 2, 5, 5 } }; /* h, w, r, pad */
  for(int t = 0; t < 4; t++)
  {
    const int h = cases[t][0], w = cases[t][1], r = cases[t][2], pad = cases[t][3], d = 2 * r + 1;
    const size_t np = (size_t)h * w;
    float *k = malloc(sizeof(float) * d * d), *ones = malloc(sizeof(float) * np);
    float *want = malloc(sizeof(float) * np), *got = malloc(sizeof(float) * np);
    fill_random(k, (size_t)d * d, 7u + (uint32_t)t);
    for(size_t i = 0; i < np; i++) ones[i] = 1.f;
    const float *planes[1] = { ones };
    float *outs[1] = { want };
    ekvido_conv_t *cv = ekvido_conv_new(planes, 1, h, w, pad, 1);
    CHECK(cv && ekvido_conv_apply(cv, k, r, outs) == 0, "coverage reference %d failed", t);
    CHECK(ekvido_conv_coverage(h, w, pad, k, r, got, 1) == 0, "coverage %d failed", t);
    CHECK(close_enough("coverage", got, want, np, 1e-6), "coverage %d (%dx%d r %d pad %d)", t, h, w, r, pad);
    CHECK(ekvido_conv_coverage(h, w, r - 1, k, r, got, 1) == -1, "coverage %d: oversize kernel accepted", t);
    ekvido_conv_free(cv);
    free(k); free(ones); free(want); free(got);
  }
}

/* The pre-Phase-1 glow, verbatim algebra: 9 basis planes, each band convolved, recoloured per pixel. */
static void ref_kazuki_glow(const float *in, int W, int H, const ekvido_kazuki_params_t *p, float diag, float clip,
                            float *glow)
{
  const size_t np = (size_t)W * H;
  float *src = malloc(sizeof(float) * np * 3), *e = malloc(sizeof(float) * np * 3), *t = malloc(sizeof(float) * np);
  unsigned char *cl = malloc(np);
  for(size_t i = 0; i < np; i++)
    for(int c = 0; c < 3; c++) src[c * np + i] = ekvido_positive(in[i * 4 + c]);
  const float *s3[3] = { src, src + np, src + 2 * np };
  float *e3[3] = { e, e + np, e + 2 * np };
  ekvido_extract_energy(s3, np, p->threshold_ev, p->knee_ev, e3);
  ekvido_warmth_map(s3, H, W, clip, (double)p->hue_radius * (double)diag, (float)exp2((double)p->threshold_ev),
                    p->neutral_point, p->warmth_softness, p->neutral_bias, t, cl);
  float *b = malloc(sizeof(float) * np * 9), *g = malloc(sizeof(float) * np * 9);
  for(size_t i = 0; i < np; i++)
  {
    const float y = ekvido_luminance(e[i], e[np + i], e[2 * np + i]);
    for(int c = 0; c < 3; c++) { b[c * np + i] = e[c * np + i]; b[(3 + c) * np + i] = t[i] * e[c * np + i]; }
    b[6 * np + i] = y; b[7 * np + i] = t[i] * y; b[8 * np + i] = t[i] * t[i] * y;
  }
  int r = 0;
  float *kernel = ekvido_build_kernel(&p->spec, diag, &r);
  const size_t kn = (size_t)(2 * r + 1) * (2 * r + 1);
  float *bands = malloc(sizeof(float) * kn * 2);
  ekvido_split_bands(kernel, r, (double)p->band_split * (double)diag, bands, bands + kn);
  const float *b9[9];
  float *g9[9];
  for(int k = 0; k < 9; k++) { b9[k] = b + k * np; g9[k] = g + k * np; }
  ekvido_conv_t *cv = ekvido_conv_new(b9, 9, H, W, r, 1);
  memset(glow, 0, sizeof(float) * np * 3);
  for(int band = 0; band < 2; band++)
  {
    ekvido_conv_apply(cv, bands + band * kn, r, g9);
    float cu[3], wu[3], du[3];
    ekvido_unit_luminance(band ? p->cool_edge : p->cool_core, cu);
    ekvido_unit_luminance(band ? p->warm_edge : p->warm_core, wu);
    for(int c = 0; c < 3; c++) du[c] = wu[c] - cu[c];
    const float pc = p->push_cool, dp = (float)((double)p->push_warm - (double)p->push_cool);
    for(size_t i = 0; i < np; i++)
      for(int c = 0; c < 3; c++)
        glow[c * np + i] += (1.f - pc) * g9[c][i] - dp * g9[3 + c][i] + cu[c] * (pc * g9[6][i] + dp * g9[7][i])
                            + du[c] * (pc * g9[7][i] + dp * g9[8][i]);
  }
  ekvido_conv_free(cv);
  free(src); free(e); free(t); free(cl); free(b); free(g); free(kernel); free(bands);
}

/* Shared synthetic frame for the Phase 1 kazuki tests: dim texture, a clipped plateau, warm and cool sources. */
static float *phase1_frame(int W, int H)
{
  const size_t np = (size_t)W * H;
  float *in = malloc(sizeof(float) * np * 4);
  for(size_t i = 0; i < np; i++)
  {
    for(int c = 0; c < 3; c++) in[i * 4 + c] = 0.05f + 0.01f * (float)((i * 5 + c) % 7);
    in[i * 4 + 3] = 1.f;
  }
  for(int y = 12; y < 15; y++)
    for(int x = 8; x < 11; x++)
      for(int c = 0; c < 3; c++) in[((size_t)y * W + x) * 4 + c] = 20.f;
  float *warm = in + ((size_t)6 * W + 30) * 4, *cool = in + ((size_t)24 * W + 33) * 4;
  warm[0] = 9.f; warm[1] = 5.f; warm[2] = 2.f;
  cool[0] = 2.f; cool[1] = 5.f; cool[2] = 9.f;
  return in;
}

static void phase1_params(ekvido_kazuki_params_t *p)
{
  memset(p, 0, sizeof(*p));
  p->strength = 0.7f; p->blend = EKVIDO_BLEND_ADDITIVE; p->knee_ev = 1.f; p->hue_radius = 0.02f;
  p->warmth_softness = 0.05f; p->neutral_bias = 0.8f; p->push_warm = 0.7f; p->push_cool = 0.7f; p->band_split = 0.006f;
  ekvido_hsv_to_rgb(40.f, 0.45f, p->warm_core); ekvido_hsv_to_rgb(22.5f, 0.88f, p->warm_edge);
  ekvido_hsv_to_rgb(193.846154f, 0.65f, p->cool_core); ekvido_hsv_to_rgb(232.173913f, 0.92f, p->cool_edge);
  p->spec = (ekvido_kernel_spec_t){ 0.006f, 1.3f, 1.85f, 1.f, 0.1f, 1.2f, 4, 0.f, 0.001f, 0.96f, 0.32f, 2.93f, 0.58f,
                                    1.f, 0.f, 0.2f };
}

/* Phase 1: frequency-domain recolouring = the old per-pixel recolouring, across pulls and colours. */
static void test_kazuki_reference_algebra(void)
{
  enum { W = 44, H = 32 };
  const size_t np = (size_t)W * H;
  float *in = phase1_frame(W, H), *out = malloc(sizeof(float) * np * 4);
  float *want = malloc(sizeof(float) * np * 3), *got = malloc(sizeof(float) * np * 3);
  const float diag = hypotf(W, H), clip = ekvido_estimate_clip_level(in, np, 4);
  const ekvido_rect_t full = { 0, 0, W, H };
  for(int v = 0; v < 4; v++)
  {
    ekvido_kazuki_params_t p;
    phase1_params(&p);
    if(v == 1) { p.push_warm = 1.f; p.push_cool = 0.f; }
    if(v == 2) { p.push_warm = 0.f; p.push_cool = 0.f; }
    if(v == 3) { ekvido_hsv_to_rgb(200.f, 0.9f, p.warm_core); ekvido_hsv_to_rgb(30.f, 0.6f, p.cool_edge); p.push_cool = 0.25f; }
    ref_kazuki_glow(in, W, H, &p, diag, clip, want);
    CHECK(ekvido_kazuki_process(in, W, H, out, full, &p, diag, clip, 0, EKVIDO_VIEW_GLOW, NULL, 1) == 0, "algebra %d: failed", v);
    for(size_t i = 0; i < np; i++)
      for(int c = 0; c < 3; c++) got[c * np + i] = out[i * 4 + c];
    CHECK(close_enough("algebra", got, want, np * 3, 1e-5), "algebra variant %d differs from the per-pixel recolouring", v);
  }
  float coef[2][3][EKVIDO_KAZUKI_NB];
  ekvido_kazuki_params_t p;
  phase1_params(&p);
  p.push_warm = 0.f; p.push_cool = 0.f;
  ekvido_kazuki_coefficients(&p, coef);
  CHECK(coef[0][0][0] == 1.f && coef[0][0][1] == 0.f && coef[1][2][2] == 1.f && coef[0][1][6] == 0.f,
        "zero pulls must pass the source colour through unchanged");
  free(in); free(out); free(want); free(got);
}

/* Phase 1 Review Focus 2, 3, 5: thread count, non-finite pixels, no energy. */
static void test_kazuki_phase1_edges(void)
{
  enum { W = 44, H = 32 };
  const size_t np = (size_t)W * H;
  float *in = phase1_frame(W, H), *a = malloc(sizeof(float) * np * 4), *b = malloc(sizeof(float) * np * 4);
  const float diag = hypotf(W, H);
  const ekvido_rect_t full = { 0, 0, W, H };
  ekvido_kazuki_params_t p;
  phase1_params(&p);
  /* non-finite pixels: no glow from them, nothing non-finite in the glow, cached = uncached */
  in[(3 * W + 3) * 4 + 0] = INFINITY; in[(20 * W + 5) * 4 + 1] = NAN; in[(28 * W + 40) * 4 + 2] = -INFINITY;
  const float clip = ekvido_estimate_clip_level(in, np, 4);
  CHECK(ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 0, EKVIDO_VIEW_GLOW, NULL, 1) == 0, "non-finite: failed");
  int finite = 1;
  for(size_t i = 0; i < np; i++)
    for(int c = 0; c < 3; c++) finite &= isfinite(a[i * 4 + c]);
  CHECK(finite, "non-finite input pixels leaked into the glow");
  ekvido_kazuki_cache_t *cache = ekvido_kazuki_cache_new();
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, cache, 1);
  ekvido_kazuki_process(in, W, H, b, full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, b, sizeof(float) * np * 4), "non-finite: cached differs from uncached");
  ekvido_kazuki_cache_free(cache);
  /* no energy: glow exactly 0, additive leaves the image untouched */
  p.threshold_ev = 30.f;
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 0, EKVIDO_VIEW_GLOW, NULL, 1);
  int zero = 1;
  for(size_t i = 0; i < np; i++)
    for(int c = 0; c < 3; c++) zero &= a[i * 4 + c] == 0.f;
  CHECK(zero, "no energy but the glow is not exactly zero");
  ekvido_kazuki_process(in, W, H, a, full, &p, diag, clip, 0, EKVIDO_VIEW_FINAL, NULL, 1);
  CHECK(!memcmp(a, in, sizeof(float) * np * 4), "no energy but additive changed the image");
  free(in); free(a); free(b);
}

/* Thread count and the cached path on a frame big enough that PocketFFT and the tiled accumulate
   really split work (44×32 barely multithreads): 1 and 4 threads, cached and uncached, export and a
   cached colour change, all the same bytes. Then the same cache on a smaller frame (new grid). */
static void test_kazuki_threads_cached(void)
{
  enum { W = 512, H = 384 };
  const size_t np = (size_t)W * H;
  float *in = phase1_frame(W, H);
  for(int n = 0; n < 24; n++) /* highlights spread over the frame, some warm, some cool */
  {
    const int cx = 20 + (n * 97) % (W - 40), cy = 20 + (n * 61) % (H - 40);
    for(int y = cy - 2; y <= cy + 2; y++)
      for(int x = cx - 2; x <= cx + 2; x++)
        for(int c = 0; c < 3; c++) in[((size_t)y * W + x) * 4 + c] = n % 2 ? 4.f + 2.f * c : 10.f - 3.f * c;
  }
  float *o[2][3];
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++) o[i][j] = malloc(sizeof(float) * np * 4);
  const float diag = hypotf(W, H), clip = ekvido_estimate_clip_level(in, np, 4);
  const ekvido_rect_t full = { 0, 0, W, H };
  ekvido_kazuki_params_t p, p2;
  phase1_params(&p);
  p2 = p;
  p2.push_warm = 0.3f; p2.warm_edge[1] = 0.8f;
#ifdef _OPENMP
  const int saved = omp_get_max_threads();
#endif
  for(int i = 0; i < 2; i++)
  {
    const int nt = i ? 4 : 1;
#ifdef _OPENMP
    omp_set_num_threads(nt);
#endif
    ekvido_kazuki_cache_t *cache = ekvido_kazuki_cache_new();
    ekvido_kazuki_process(in, W, H, o[i][0], full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, cache, nt);
    ekvido_kazuki_process(in, W, H, o[i][1], full, &p2, diag, clip, 1, EKVIDO_VIEW_FINAL, cache, nt);
    ekvido_kazuki_cache_free(cache);
    ekvido_kazuki_process(in, W, H, o[i][2], full, &p2, diag, clip, 1, EKVIDO_VIEW_FINAL, NULL, nt);
    CHECK(!memcmp(o[i][1], o[i][2], sizeof(float) * np * 4), "%d threads: cached colour change differs from uncached", nt);
  }
#ifdef _OPENMP
  omp_set_num_threads(saved);
#endif
  CHECK(!memcmp(o[0][0], o[1][0], sizeof(float) * np * 4), "cached kazuki output depends on the thread count");
  CHECK(!memcmp(o[0][1], o[1][1], sizeof(float) * np * 4), "cached colour change depends on the thread count");
  CHECK(!memcmp(o[0][2], o[1][2], sizeof(float) * np * 4), "uncached kazuki output depends on the thread count");
  int changed = 0;
  for(size_t q = 0; q < np * 4; q++) changed |= o[0][0][q] != o[0][1][q];
  CHECK(changed, "colour change did not change the output");
  /* one cache across a frame-size change: scratch rebuilt on the new grid */
  {
    enum { w = 44, h = 32 };
    const size_t n2 = (size_t)w * h;
    float *small = phase1_frame(w, h);
    const float d2 = hypotf(w, h), c2 = ekvido_estimate_clip_level(small, n2, 4);
    const ekvido_rect_t f2 = { 0, 0, w, h };
    ekvido_kazuki_cache_t *cache = ekvido_kazuki_cache_new();
    ekvido_kazuki_process(in, W, H, o[0][0], full, &p, diag, clip, 1, EKVIDO_VIEW_FINAL, cache, 1);
    ekvido_kazuki_process(small, w, h, o[0][1], f2, &p2, d2, c2, 1, EKVIDO_VIEW_FINAL, cache, 1);
    ekvido_kazuki_process(small, w, h, o[0][2], f2, &p2, d2, c2, 1, EKVIDO_VIEW_FINAL, NULL, 1);
    CHECK(!memcmp(o[0][1], o[0][2], sizeof(float) * n2 * 4), "cache reused across a frame-size change differs from uncached");
    ekvido_kazuki_cache_free(cache);
    free(small);
  }
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++) free(o[i][j]);
  free(in);
}

/* @@CASE_HANDLERS: later tasks add static void case_<kind>(const char *line, const char *name) here */

static void run_cases(void)
{
  char path[1024], line[4096];
  snprintf(path, sizeof(path), "%s/cases.txt", g_dir);
  FILE *f = fopen(path, "r");
  CHECK(f != NULL, "cannot open %s", path);
  if(!f) return;
  while(fgets(line, sizeof(line), f))
  {
    char kind[32], name[128];
    line[strcspn(line, "\n")] = 0;
    if(sscanf(line, "%31s %127s", kind, name) != 2) continue;
    if(!strcmp(kind, "psf")) case_psf(line, name);
    if(!strcmp(kind, "conv")) case_conv(line, name);
    if(!strcmp(kind, "mist")) case_mist(line, name);
    if(!strcmp(kind, "kview")) case_kview(line, name);
    if(!strcmp(kind, "energy")) case_energy(line, name);
    if(!strcmp(kind, "clip")) case_clip(line, name);
    if(!strcmp(kind, "blur")) case_blur(line, name);
    if(!strcmp(kind, "wblur")) case_wblur(line, name);
        if(!strcmp(kind, "blend")) case_blend(line, name);
    if(!strcmp(kind, "warmth")) case_warmth(line, name);
    if(!strcmp(kind, "bands")) case_bands(line, name);
    if(!strcmp(kind, "hsv")) case_hsv(line, name);
    if(!strcmp(kind, "kazuki")) case_kazuki(line, name);
    if(!strcmp(kind, "maskenergy")) case_maskenergy(line, name);
  /* @@CASE_DISPATCH: later tasks add  if(!strcmp(kind, "<kind>")) case_<kind>(line, name); */
  }
  fclose(f);
}

int main(int argc, char **argv)
{
  if(argc > 1) g_dir = argv[1];
  test_fft_size();
  test_fft_round_trip();
  test_psf_extremes();
  test_kernel_threads();
  test_coverage();
  test_grid_matches_conv();
  test_coverage_sat();
  test_expand_rect();
  test_mist_identity_and_cache();
  test_kernel_view();
    test_blend_formulas();
  test_warmth_colour();
  test_kazuki_cache_views_identity();
  test_kazuki_source_mask();
  test_kazuki_reference_algebra();
  test_kazuki_phase1_edges();
  test_kazuki_threads_cached();
  /* @@UNIT_TESTS: later tasks add calls here */
  run_cases();
  printf("%d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
