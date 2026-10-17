/* ekvido: OpenCL twins of the C core (src/common/ekvido/), one kernel per CPU loop; the CPU code
   is the reference and every expression keeps its operand order. Built by ekvido itself
   (src/common/ekvido/cl/ekv_cl.c), not listed in programs.conf, never with
   -cl-fast-relaxed-math; non-finite values are classified by bit pattern. */

#define EKV_Y_R 0.2627f
#define EKV_Y_G 0.6780f
#define EKV_Y_B 0.0593f
#define EKV_GLOW_EPS 1e-6f

constant sampler_t ekv_px = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

static inline int ekv_finite(const float v) { return (as_uint(v) & 0x7f800000u) != 0x7f800000u; }
static inline int ekv_nan(const float v) { return (as_uint(v) & 0x7fffffffu) > 0x7f800000u; }
static inline float ekv_positive(const float v) { return ekv_finite(v) && v > 0.f ? v : 0.f; }
static inline float ekv_lum(const float r, const float g, const float b) { return r * EKV_Y_R + g * EKV_Y_G + b * EKV_Y_B; }

kernel void ekv_zero(global float *b, const uint n)
{
  const uint i = get_global_id(0);
  if(i < n) b[i] = 0.f;
}

/* ekvido_smoothstep with lo = (float)lo and span = (float)(hi - lo) from the host; flat: hi <= lo */
static inline float ekv_smoothstep(const float x, const float lo, const float span, const int flat)
{
  if(flat) return x >= lo ? 1.f : 0.f;
  float t = (x - lo) / span;
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  return t * t * (3.f - 2.f * t);
}

static inline float ekv_max3(const float4 p)
{
  return fmax(fmax(ekv_positive(p.x), ekv_positive(p.y)), ekv_positive(p.z));
}

kernel void ekv_copy(global const float *in, global float *out, const uint n)
{
  const uint i = get_global_id(0);
  if(i < n) out[i] = in[i];
}

/* _source (kazuki.c, mist.c): positive RGB planes */
kernel void ekv_split(read_only image2d_t in, const int w, const int h, global float *src)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= w || y >= h) return;
  const float4 p = read_imagef(in, ekv_px, (int2)(x, y));
  const size_t np = (size_t)w * h, i = (size_t)y * w + x;
  src[i] = ekv_positive(p.x);
  src[np + i] = ekv_positive(p.y);
  src[2 * np + i] = ekv_positive(p.z);
}

/* ekvido_extract_energy */
kernel void ekv_energy(global const float *src, global float *e, const uint np, const float lo, const float span,
                       const int flat)
{
  const uint i = get_global_id(0);
  if(i >= np) return;
  const float y = ekv_lum(src[i], src[(size_t)np + i], src[2 * (size_t)np + i]);
  const float k = ekv_smoothstep(log2(fmax(y, 1e-8f)), lo, span, flat);
  for(int c = 0; c < 3; c++) e[c * (size_t)np + i] = src[c * (size_t)np + i] * k;
}

/* ekvido_mask_energy */
kernel void ekv_mask_energy(global float *e, global const float *mask, const uint np, const int invert)
{
  const uint i = get_global_id(0);
  if(i >= np) return;
  float m = fmin(ekv_positive(mask[i]), 1.f);
  if(invert) m = 1.f - m;
  for(int c = 0; c < 3; c++) e[c * (size_t)np + i] *= m;
}

/* ekvido_estimate_clip_level, per row (the host finishes): max channel, then the plateau count */
kernel void ekv_row_max(read_only image2d_t in, const int w, const int h, global float *part)
{
  const int y = get_global_id(0);
  if(y >= h) return;
  float top = 0.f;
  for(int x = 0; x < w; x++) top = fmax(top, ekv_max3(read_imagef(in, ekv_px, (int2)(x, y))));
  part[y] = top;
}

kernel void ekv_row_count(read_only image2d_t in, const int w, const int h, const float near, global uint *part)
{
  const int y = get_global_id(0);
  if(y >= h) return;
  uint n = 0;
  for(int x = 0; x < w; x++) n += ekv_max3(read_imagef(in, ekv_px, (int2)(x, y))) >= near;
  part[y] = n;
}

/* FNV-1a over a row's 32-bit words (GPU cache key; not the CPU's ekvido_hash_floats) */
kernel void ekv_row_hash(read_only image2d_t in, const int w, const int h, global ulong *part)
{
  const int y = get_global_id(0);
  if(y >= h) return;
  ulong hh = 1469598103934665603UL;
  for(int x = 0; x < w; x++)
  {
    const uint4 u = as_uint4(read_imagef(in, ekv_px, (int2)(x, y)));
    hh ^= u.x; hh *= 1099511628211UL;
    hh ^= u.y; hh *= 1099511628211UL;
    hh ^= u.z; hh *= 1099511628211UL;
    hh ^= u.w; hh *= 1099511628211UL;
  }
  part[y] = hh;
}

/* ekvido_warmth_map, first loop */
kernel void ekv_warmth_prep(global const float *src, const uint np, const float clip_level, global float *wt,
                            global float *xr, global float *xb, global float *cm, global uchar *clipped)
{
  const uint i = get_global_id(0);
  if(i >= np) return;
  const float r = src[i], g = src[(size_t)np + i], b = src[2 * (size_t)np + i];
  const int cl = fmax(fmax(r, g), b) >= clip_level;
  const float s = r + g + b;
  clipped[i] = (uchar)cl;
  wt[i] = cl ? 0.f : ekv_lum(r, g, b);
  xr[i] = s > 1e-12f ? r / fmax(s, 1e-12f) : 1.f / 3.f;
  xb[i] = s > 1e-12f ? b / fmax(s, 1e-12f) : 1.f / 3.f;
  cm[i] = cl ? 1.f : 0.f;
}

/* ekvido_warmth_map, last loop: confidence smoothstep(conf, 0.15, 0.4) has lo 0.15f, span 0.25f */
kernel void ekv_warmth_finish(global const float *den, global const float *cm, global const float *hr,
                              global const float *hb, const uint np, const float y_floor, const float neutral_point,
                              const float soft_lo, const float soft_span, const int soft_flat, const float bias,
                              global float *t)
{
  const uint i = get_global_id(0);
  if(i >= np) return;
  const float conf = den[i] / (den[i] + cm[i] * y_floor + 1e-12f);
  const float a = ekv_smoothstep(conf, 0.15f, 0.25f, 0);
  const float warm = ekv_smoothstep(hr[i] - hb[i] - neutral_point, soft_lo, soft_span, soft_flat);
  t[i] = a * warm + (1.f - a) * bias;
}

/* wblur.c _direct: horizontal then vertical, replicate edges; taps from the host */
kernel void ekv_blur_h(global const float *in, global float *out, const int h, const int w, constant float *g,
                       const int r)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= w || y >= h) return;
  float acc = 0.f;
  for(int t = 0; t < 2 * r + 1; t++) acc += g[t] * in[(size_t)y * w + clamp(x + t - r, 0, w - 1)];
  out[(size_t)y * w + x] = acc;
}

kernel void ekv_blur_v(global const float *in, global float *out, const int h, const int w, constant float *g,
                       const int r)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= w || y >= h) return;
  float acc = 0.f;
  for(int t = 0; t < 2 * r + 1; t++) acc += g[t] * in[(size_t)clamp(y + t - r, 0, h - 1) * w + x];
  out[(size_t)y * w + x] = acc;
}

/* wblur.c _downsampled: box-downsample by f, replicate-padded */
kernel void ekv_box_down(global const float *in, global float *small, const int h, const int w, const int f,
                         const int sh, const int sw, const float inv)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= sw || y >= sh) return;
  float acc = 0.f;
  for(int dy = 0; dy < f; dy++)
    for(int dx = 0; dx < f; dx++)
      acc += in[(size_t)clamp(y * f + dy, 0, h - 1) * w + clamp(x * f + dx, 0, w - 1)];
  small[(size_t)y * sw + x] = acc * inv;
}

/* PyTorch bilinear, align_corners = False */
kernel void ekv_bilinear_up(global const float *small, global float *out, const int h, const int w, const int sh,
                            const int sw, const float scale_y, const float scale_x)
{
  const int j = get_global_id(0), i = get_global_id(1);
  if(j >= w || i >= h) return;
  const float sy = fmax(scale_y * ((float)i + 0.5f) - 0.5f, 0.f);
  const int y0 = (int)sy, y1 = y0 + (y0 < sh - 1 ? 1 : 0);
  const float ly1 = sy - (float)y0, ly0 = 1.f - ly1;
  const float sx = fmax(scale_x * ((float)j + 0.5f) - 0.5f, 0.f);
  const int x0 = (int)sx, x1 = x0 + (x0 < sw - 1 ? 1 : 0);
  const float lx1 = sx - (float)x0, lx0 = 1.f - lx1;
  global const float *r0 = small + (size_t)y0 * sw, *r1 = small + (size_t)y1 * sw;
  out[(size_t)i * w + j] = ly0 * (lx0 * r0[x0] + lx1 * r0[x1]) + ly1 * (lx0 * r1[x0] + lx1 * r1[x1]);
}

/* ekvido_weighted_blur loops */
kernel void ekv_mul(global const float *x, global const float *wt, global float *out, const uint n)
{
  const uint i = get_global_id(0);
  if(i < n) out[i] = x[i] * wt[i];
}

kernel void ekv_div_floor(global float *out, global const float *den, const uint n)
{
  const uint i = get_global_id(0);
  if(i < n) out[i] /= fmax(den[i], 1e-12f);
}

/* ekvido_kazuki_basis_plane (k) into the zero-padded in-place R2C grid (rows of `stride` floats);
   with e = the source planes and k < 3 this is mist's plane k. t is not read for k < 3. */
kernel void ekv_basis_grid(global const float *e, global const float *t, const int w, const int h, const int k,
                           global float *grid, const int fh, const int stride)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= stride || y >= fh) return;
  float v = 0.f;
  if(x < w && y < h)
  {
    const size_t np = (size_t)w * h, i = (size_t)y * w + x;
    if(k < 3)
      v = e[k * np + i];
    else if(k < 6)
      v = t[i] * e[(k - 3) * np + i];
    else
      v = t[i] * t[i] * ekv_lum(e[i], e[np + i], e[2 * np + i]);
  }
  grid[(size_t)y * stride + x] = v;
}

/* ekvido_grid_kernel_spectrum's placement: (2r+1)² kernel centred at the origin, wrapped */
kernel void ekv_place(global const float *kern, const int r, global float *grid, const int fh, const int fw,
                      const int stride)
{
  const int j = get_global_id(0), i = get_global_id(1), d = 2 * r + 1;
  if(i >= d || j >= d) return;
  grid[(size_t)((i - r + fh) % fh) * stride + (size_t)((j - r + fw) % fw)] = kern[(size_t)i * d + j];
}

static inline void ekv_mac1(global float *ac, const size_t q, const int k, const float ci, const float co,
                            const float ir, const float ii, const float orr, const float oi)
{
  const float re = ci * ir + co * orr, im = ci * ii + co * oi;
  ac[q] = k ? ac[q] + re : re;
  ac[q + 1] = k ? ac[q + 1] + im : im;
}

/* _mac_tile (kazuki.c) for one plane k: acc_c (=, k == 0 / +=) S·(ci_c·Ki + co_c·Ko) */
kernel void ekv_mac(global const float *s, global const float *ki, global const float *ko, const uint nc, const int k,
                    const float ci0, const float ci1, const float ci2, const float co0, const float co1,
                    const float co2, global float *acc0, global float *acc1, global float *acc2)
{
  const uint m = get_global_id(0);
  if(m >= nc) return;
  const size_t q = 2 * (size_t)m;
  const float a = s[q], b = s[q + 1];
  const float ir = a * ki[q] - b * ki[q + 1], ii = a * ki[q + 1] + b * ki[q];
  const float orr = a * ko[q] - b * ko[q + 1], oi = a * ko[q + 1] + b * ko[q];
  ekv_mac1(acc0, q, k, ci0, co0, ir, ii, orr, oi);
  ekv_mac1(acc1, q, k, ci1, co1, ir, ii, orr, oi);
  ekv_mac1(acc2, q, k, ci2, co2, ir, ii, orr, oi);
}

/* ekvido_conv_apply's spectrum product, in place */
kernel void ekv_cmul(global float *s, global const float *kf, const uint nc)
{
  const uint m = get_global_id(0);
  if(m >= nc) return;
  const size_t q = 2 * (size_t)m;
  const float a = s[q], b = s[q + 1], e = kf[q], f = kf[q + 1];
  s[q] = a * e - b * f;
  s[q + 1] = a * f + b * e;
}

/* ekvido_grid_inverse's crop, with the 1/(fh·fw) normalisation fft.cc applies */
kernel void ekv_crop(global const float *grid, const int stride, const int w, const int h, const float scale,
                     global float *out, const uint off)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= w || y >= h) return;
  out[off + (size_t)y * w + x] = grid[(size_t)y * stride + x] * scale;
}

/* mist.c _glow: glow /= max(coverage, 1e-6) */
kernel void ekv_div_cover(global float *glow, global const float *cover, const uint np)
{
  const uint i = get_global_id(0);
  if(i >= np) return;
  const float inv = 1.f / fmax(cover[i], 1e-6f);
  for(int c = 0; c < 3; c++) glow[c * (size_t)np + i] *= inv;
}

/* ---- blend.c / color.c twins ---- */
static inline float ekv_clamp01(const float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

static inline float ekv_srgb_encode(float x)
{
  x = ekv_clamp01(x);
  return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

static inline float ekv_srgb_decode(float v)
{
  v = ekv_clamp01(v);
  return v <= 0.04045f ? v / 12.92f : pow((v + 0.055f) / 1.055f, 2.4f);
}

static inline float ekv_dodge(const float a, const float c)
{
  if(a <= 0.f) return 0.f;
  if(c >= 1.f) return 1.f;
  if(ekv_nan(a) || ekv_nan(c)) return NAN;
  return fmin(1.f, a / fmax(1.f - c, 1e-12f));
}

static inline float ekv_overlay(const float a, const float c)
{
  return a <= 0.5f ? 2.f * a * c : 1.f - 2.f * (1.f - a) * (1.f - c);
}

static inline float ekv_overlay_glow(const float a, const float c, const float m, const float alpha)
{
  if(!(m > EKV_GLOW_EPS)) return a;
  return a + alpha * m * (ekv_overlay(a, fmin(c / m, 1.f)) - a);
}

/* modes as ekvido_blend_mode_t: 0 additive, 1 conserve, 2 screen, 3 overlay, 4 colour dodge,
   5 glow dodge, 6 overlay (glow only) */
static inline float ekv_csp_blend(const int mode, const float a, const float c, const float alpha)
{
  switch(mode)
  {
    case 2: return a + alpha * ((1.f - (1.f - a) * (1.f - c)) - a);
    case 3: return a + alpha * (ekv_overlay(a, c) - a);
    case 4: return a + alpha * (ekv_dodge(a, c) - a);
    case 5: return ekv_dodge(a, alpha * c);
    case 6: return ekv_overlay_glow(a, c, c, alpha);
    default: return a;
  }
}

static inline float ekv_blend_encoded(const int mode, const float img, const float big_c, const float m,
                                      const float alpha, const float white)
{
  const float a = img / white;
  const float a_c = ekv_clamp01(a);
  const float excess = a - a_c;
  const float big_a = ekv_srgb_encode(a_c);
  const float r = mode == 6 ? ekv_overlay_glow(big_a, big_c, m, alpha) : ekv_csp_blend(mode, big_a, big_c, alpha);
  return white * (ekv_srgb_decode(r) + excess);
}

static inline float ekv_blend(const int mode, const float img, const float glow, const float energy,
                              const float strength, const float white)
{
  if(mode == 0) return img + strength * glow;
  const float alpha = fmin(strength, 1.f);
  if(mode == 1) return img + alpha * (glow - energy);
  if(alpha == 0.f || mode == 6) return img;
  return ekv_blend_encoded(mode, img, ekv_srgb_encode(fmin(glow / white, 1.f)), 0.f, alpha, white);
}

static inline void ekv_blend_px(const int mode, const float *img, const float *glow, const float *energy,
                                const float strength, const float white, float *out)
{
  if(mode != 6)
  {
    for(int c = 0; c < 3; c++) out[c] = ekv_blend(mode, img[c], glow[c], energy[c], strength, white);
    return;
  }
  const float alpha = fmin(strength, 1.f);
  float big_c[3], m = 0.f;
  for(int c = 0; c < 3; c++)
  {
    big_c[c] = ekv_srgb_encode(fmin(glow[c] / white, 1.f));
    m = fmax(m, big_c[c]);
  }
  if(alpha == 0.f || !(m > EKV_GLOW_EPS))
  {
    for(int c = 0; c < 3; c++) out[c] = img[c];
    return;
  }
  for(int c = 0; c < 3; c++) out[c] = ekv_blend_encoded(mode, img[c], big_c[c], m, alpha, white);
}

constant float ekv_cool[3] = { 40.f, 90.f, 255.f }, ekv_mid[3] = { 128.f, 128.f, 128.f },
               ekv_warm[3] = { 255.f, 140.f, 30.f };

static inline void ekv_warmth_colour(float t, float *rgb)
{
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  for(int c = 0; c < 3; c++)
  {
    const float v = t <= 0.5f ? ekv_cool[c] + (ekv_mid[c] - ekv_cool[c]) * (t * 2.f)
                              : ekv_mid[c] + (ekv_warm[c] - ekv_mid[c]) * (t * 2.f - 1.f);
    rgb[c] = ekv_srgb_decode(rint(v) / 255.f);
  }
}

/* ekvido_kazuki_process_masked's per-pixel stage; views as ekvido_kazuki_view_t */
kernel void ekv_kazuki_out(read_only image2d_t in, write_only image2d_t out, const int rx, const int ry, const int rw,
                           const int rh, const int iw, const uint np, const int view, global const float *glow,
                           global const float *energy, global const float *t, global const uchar *clipped,
                           const int mode, const float strength, const float white)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= rw || y >= rh) return;
  const int sx = rx + x, sy = ry + y;
  const float4 p = read_imagef(in, ekv_px, (int2)(sx, sy));
  const size_t si = (size_t)sy * iw + sx;
  const float s[3] = { p.x, p.y, p.z };
  float o[3];
  switch(view)
  {
    case 1:
      for(int c = 0; c < 3; c++) o[c] = glow[c * (size_t)np + si];
      break;
    case 2:
      for(int c = 0; c < 3; c++) o[c] = energy[c * (size_t)np + si];
      break;
    case 3:
      ekv_warmth_colour(t[si], o);
      break;
    case 4:
      for(int c = 0; c < 3; c++) o[c] = clipped[si] ? (c == 0 ? 1.f : 0.f) : s[c];
      break;
    default:
    {
      const float gl[3] = { glow[si], glow[(size_t)np + si], glow[2 * (size_t)np + si] };
      const float e[3] = { energy[si], energy[(size_t)np + si], energy[2 * (size_t)np + si] };
      ekv_blend_px(mode, s, gl, e, strength, white, o);
    }
  }
  write_imagef(out, (int2)(x, y), (float4)(o[0], o[1], o[2], p.w));
}

/* ekvido_mist_process's per-pixel stage (non-finite inputs spelled out, as in mist.c) */
kernel void ekv_mist_out(read_only image2d_t in, write_only image2d_t out, const int rx, const int ry, const int rw,
                         const int rh, const int iw, const uint np, global const float *glow, const float k,
                         const int glow_only)
{
  const int x = get_global_id(0), y = get_global_id(1);
  if(x >= rw || y >= rh) return;
  const int sx = rx + x, sy = ry + y;
  const float4 p = read_imagef(in, ekv_px, (int2)(sx, sy));
  const size_t si = (size_t)sy * iw + sx;
  const float s[3] = { p.x, p.y, p.z };
  float o[3];
  for(int c = 0; c < 3; c++)
  {
    const float g = glow[c * (size_t)np + si];
    o[c] = glow_only ? g : ekv_finite(s[c]) ? (1.f - k) * s[c] + k * g : (k < 1.f ? s[c] : NAN);
  }
  write_imagef(out, (int2)(x, y), (float4)(o[0], o[1], o[2], p.w));
}
