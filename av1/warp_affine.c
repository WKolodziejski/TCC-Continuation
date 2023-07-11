#include <bits/types.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>
#include "corner_match.h"

typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;

#define AOMMIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX_PARAMDIM 9
#define WARP_ERROR_BLOCK_LOG 5
#define WARP_ERROR_BLOCK (1 << WARP_ERROR_BLOCK_LOG)
#define WARPEDMODEL_PREC_BITS 16
#define ERRORADV_BORDER 0
#define DIV_LUT_PREC_BITS 14
#define DIV_LUT_BITS 8
#define DIV_LUT_NUM (1 << DIV_LUT_BITS)
#define ROUND_POWER_OF_TWO_64(value, n) \
  (((value) + ((((int64_t)1 << (n)) >> 1))) >> (n))
#define ROUND_POWER_OF_TWO(value, n) (((value) + (((1 << (n)) >> 1))) >> (n))
#define ROUND_POWER_OF_TWO_SIGNED_64(value, n)           \
  (((value) < 0) ? -ROUND_POWER_OF_TWO_64(-(value), (n)) \
                 : ROUND_POWER_OF_TWO_64((value), (n)))
#define ROUND_POWER_OF_TWO_SIGNED(value, n)           \
  (((value) < 0) ? -ROUND_POWER_OF_TWO(-(value), (n)) \
                 : ROUND_POWER_OF_TWO((value), (n)))
#define IMPLIES(a, b) (!(a) || (b))
#define WARPEDPIXEL_PREC_BITS 6
#define WARPEDPIXEL_PREC_SHIFTS (1 << WARPEDPIXEL_PREC_BITS)
#define WARP_PARAM_REDUCE_BITS 6
#define WARPEDDIFF_PREC_BITS (WARPEDMODEL_PREC_BITS - WARPEDPIXEL_PREC_BITS)
#define FILTER_BITS 7
#define DIST_PRECISION_BITS 4
#define DECLARE_ALIGNED(n, typ, val) typ val
#define ROUND0_BITS 3
#define COMPOUND_ROUND1_BITS 7
#define GM_REFINEMENT_COUNT 5
#define GM_TRANS_PREC_BITS 6
#define GM_ABS_TRANS_BITS 12
#define GM_TRANS_PREC_DIFF (WARPEDMODEL_PREC_BITS - GM_TRANS_PREC_BITS)
#define GM_ALPHA_PREC_BITS 15
#define GM_ABS_ALPHA_BITS 12
#define GM_ALPHA_PREC_DIFF (WARPEDMODEL_PREC_BITS - GM_ALPHA_PREC_BITS)
#define WARPEDMODEL_ROW3HOMO_PREC_BITS 16
#define GM_ROW3HOMO_PREC_BITS 16
#define GM_ABS_ROW3HOMO_BITS 11
#define GM_ROW3HOMO_PREC_DIFF \
  (WARPEDMODEL_ROW3HOMO_PREC_BITS - GM_ROW3HOMO_PREC_BITS)
#define GM_TRANS_MAX (1 << GM_ABS_TRANS_BITS)
#define GM_ALPHA_MAX (1 << GM_ABS_ALPHA_BITS)
#define GM_ROW3HOMO_MAX (1 << GM_ABS_ROW3HOMO_BITS)
#define GM_ROW3HOMO_PREC_BITS 16
#define GM_ABS_ROW3HOMO_BITS 11
#define GM_ROW3HOMO_PREC_DIFF \
  (WARPEDMODEL_ROW3HOMO_PREC_BITS - GM_ROW3HOMO_PREC_BITS)
# define INT8_MAX		(127)
# define INT16_MAX		(32767)
# define INT32_MAX		(2147483647)
# define INT64_MAX		(__INT64_C(9223372036854775807))
# define INT8_MIN		(-128)
# define INT16_MIN		(-32767-1)
# define INT32_MIN		(-2147483647-1)
# define INT64_MIN		(-__INT64_C(9223372036854775807)-1)

typedef struct WarpedMotionParams {
  int32_t wmmat[6];
  int16_t alpha, beta, gamma, delta;
  enum TransformationType wmtype;
  int8_t invalid;
} WarpedMotionParams;

typedef uint16_t CONV_BUF_TYPE;
typedef struct ConvolveParams {
  int do_average;
  CONV_BUF_TYPE *dst;
  int dst_stride;
  int round_0;
  int round_1;
  int plane;
  int is_compound;
  int use_dist_wtd_comp_avg;
  int fwd_offset;
  int bck_offset;
} ConvolveParams;

const int16_t av1_warped_filter[WARPEDPIXEL_PREC_SHIFTS * 3 + 1][8] = {
  // [-1, 0)
  { 0,   0, 127,   1,   0, 0, 0, 0 }, { 0, - 1, 127,   2,   0, 0, 0, 0 },
  { 1, - 3, 127,   4, - 1, 0, 0, 0 }, { 1, - 4, 126,   6, - 2, 1, 0, 0 },
  { 1, - 5, 126,   8, - 3, 1, 0, 0 }, { 1, - 6, 125,  11, - 4, 1, 0, 0 },
  { 1, - 7, 124,  13, - 4, 1, 0, 0 }, { 2, - 8, 123,  15, - 5, 1, 0, 0 },
  { 2, - 9, 122,  18, - 6, 1, 0, 0 }, { 2, -10, 121,  20, - 6, 1, 0, 0 },
  { 2, -11, 120,  22, - 7, 2, 0, 0 }, { 2, -12, 119,  25, - 8, 2, 0, 0 },
  { 3, -13, 117,  27, - 8, 2, 0, 0 }, { 3, -13, 116,  29, - 9, 2, 0, 0 },
  { 3, -14, 114,  32, -10, 3, 0, 0 }, { 3, -15, 113,  35, -10, 2, 0, 0 },
  { 3, -15, 111,  37, -11, 3, 0, 0 }, { 3, -16, 109,  40, -11, 3, 0, 0 },
  { 3, -16, 108,  42, -12, 3, 0, 0 }, { 4, -17, 106,  45, -13, 3, 0, 0 },
  { 4, -17, 104,  47, -13, 3, 0, 0 }, { 4, -17, 102,  50, -14, 3, 0, 0 },
  { 4, -17, 100,  52, -14, 3, 0, 0 }, { 4, -18,  98,  55, -15, 4, 0, 0 },
  { 4, -18,  96,  58, -15, 3, 0, 0 }, { 4, -18,  94,  60, -16, 4, 0, 0 },
  { 4, -18,  91,  63, -16, 4, 0, 0 }, { 4, -18,  89,  65, -16, 4, 0, 0 },
  { 4, -18,  87,  68, -17, 4, 0, 0 }, { 4, -18,  85,  70, -17, 4, 0, 0 },
  { 4, -18,  82,  73, -17, 4, 0, 0 }, { 4, -18,  80,  75, -17, 4, 0, 0 },
  { 4, -18,  78,  78, -18, 4, 0, 0 }, { 4, -17,  75,  80, -18, 4, 0, 0 },
  { 4, -17,  73,  82, -18, 4, 0, 0 }, { 4, -17,  70,  85, -18, 4, 0, 0 },
  { 4, -17,  68,  87, -18, 4, 0, 0 }, { 4, -16,  65,  89, -18, 4, 0, 0 },
  { 4, -16,  63,  91, -18, 4, 0, 0 }, { 4, -16,  60,  94, -18, 4, 0, 0 },
  { 3, -15,  58,  96, -18, 4, 0, 0 }, { 4, -15,  55,  98, -18, 4, 0, 0 },
  { 3, -14,  52, 100, -17, 4, 0, 0 }, { 3, -14,  50, 102, -17, 4, 0, 0 },
  { 3, -13,  47, 104, -17, 4, 0, 0 }, { 3, -13,  45, 106, -17, 4, 0, 0 },
  { 3, -12,  42, 108, -16, 3, 0, 0 }, { 3, -11,  40, 109, -16, 3, 0, 0 },
  { 3, -11,  37, 111, -15, 3, 0, 0 }, { 2, -10,  35, 113, -15, 3, 0, 0 },
  { 3, -10,  32, 114, -14, 3, 0, 0 }, { 2, - 9,  29, 116, -13, 3, 0, 0 },
  { 2, - 8,  27, 117, -13, 3, 0, 0 }, { 2, - 8,  25, 119, -12, 2, 0, 0 },
  { 2, - 7,  22, 120, -11, 2, 0, 0 }, { 1, - 6,  20, 121, -10, 2, 0, 0 },
  { 1, - 6,  18, 122, - 9, 2, 0, 0 }, { 1, - 5,  15, 123, - 8, 2, 0, 0 },
  { 1, - 4,  13, 124, - 7, 1, 0, 0 }, { 1, - 4,  11, 125, - 6, 1, 0, 0 },
  { 1, - 3,   8, 126, - 5, 1, 0, 0 }, { 1, - 2,   6, 126, - 4, 1, 0, 0 },
  { 0, - 1,   4, 127, - 3, 1, 0, 0 }, { 0,   0,   2, 127, - 1, 0, 0, 0 },

  // [0, 1)
  { 0,  0,   0, 127,   1,   0,  0,  0}, { 0,  0,  -1, 127,   2,   0,  0,  0},
  { 0,  1,  -3, 127,   4,  -2,  1,  0}, { 0,  1,  -5, 127,   6,  -2,  1,  0},
  { 0,  2,  -6, 126,   8,  -3,  1,  0}, {-1,  2,  -7, 126,  11,  -4,  2, -1},
  {-1,  3,  -8, 125,  13,  -5,  2, -1}, {-1,  3, -10, 124,  16,  -6,  3, -1},
  {-1,  4, -11, 123,  18,  -7,  3, -1}, {-1,  4, -12, 122,  20,  -7,  3, -1},
  {-1,  4, -13, 121,  23,  -8,  3, -1}, {-2,  5, -14, 120,  25,  -9,  4, -1},
  {-1,  5, -15, 119,  27, -10,  4, -1}, {-1,  5, -16, 118,  30, -11,  4, -1},
  {-2,  6, -17, 116,  33, -12,  5, -1}, {-2,  6, -17, 114,  35, -12,  5, -1},
  {-2,  6, -18, 113,  38, -13,  5, -1}, {-2,  7, -19, 111,  41, -14,  6, -2},
  {-2,  7, -19, 110,  43, -15,  6, -2}, {-2,  7, -20, 108,  46, -15,  6, -2},
  {-2,  7, -20, 106,  49, -16,  6, -2}, {-2,  7, -21, 104,  51, -16,  7, -2},
  {-2,  7, -21, 102,  54, -17,  7, -2}, {-2,  8, -21, 100,  56, -18,  7, -2},
  {-2,  8, -22,  98,  59, -18,  7, -2}, {-2,  8, -22,  96,  62, -19,  7, -2},
  {-2,  8, -22,  94,  64, -19,  7, -2}, {-2,  8, -22,  91,  67, -20,  8, -2},
  {-2,  8, -22,  89,  69, -20,  8, -2}, {-2,  8, -22,  87,  72, -21,  8, -2},
  {-2,  8, -21,  84,  74, -21,  8, -2}, {-2,  8, -22,  82,  77, -21,  8, -2},
  {-2,  8, -21,  79,  79, -21,  8, -2}, {-2,  8, -21,  77,  82, -22,  8, -2},
  {-2,  8, -21,  74,  84, -21,  8, -2}, {-2,  8, -21,  72,  87, -22,  8, -2},
  {-2,  8, -20,  69,  89, -22,  8, -2}, {-2,  8, -20,  67,  91, -22,  8, -2},
  {-2,  7, -19,  64,  94, -22,  8, -2}, {-2,  7, -19,  62,  96, -22,  8, -2},
  {-2,  7, -18,  59,  98, -22,  8, -2}, {-2,  7, -18,  56, 100, -21,  8, -2},
  {-2,  7, -17,  54, 102, -21,  7, -2}, {-2,  7, -16,  51, 104, -21,  7, -2},
  {-2,  6, -16,  49, 106, -20,  7, -2}, {-2,  6, -15,  46, 108, -20,  7, -2},
  {-2,  6, -15,  43, 110, -19,  7, -2}, {-2,  6, -14,  41, 111, -19,  7, -2},
  {-1,  5, -13,  38, 113, -18,  6, -2}, {-1,  5, -12,  35, 114, -17,  6, -2},
  {-1,  5, -12,  33, 116, -17,  6, -2}, {-1,  4, -11,  30, 118, -16,  5, -1},
  {-1,  4, -10,  27, 119, -15,  5, -1}, {-1,  4,  -9,  25, 120, -14,  5, -2},
  {-1,  3,  -8,  23, 121, -13,  4, -1}, {-1,  3,  -7,  20, 122, -12,  4, -1},
  {-1,  3,  -7,  18, 123, -11,  4, -1}, {-1,  3,  -6,  16, 124, -10,  3, -1},
  {-1,  2,  -5,  13, 125,  -8,  3, -1}, {-1,  2,  -4,  11, 126,  -7,  2, -1},
  { 0,  1,  -3,   8, 126,  -6,  2,  0}, { 0,  1,  -2,   6, 127,  -5,  1,  0},
  { 0,  1,  -2,   4, 127,  -3,  1,  0}, { 0,  0,   0,   2, 127,  -1,  0,  0},

  // [1, 2)
  { 0, 0, 0,   1, 127,   0,   0, 0 }, { 0, 0, 0, - 1, 127,   2,   0, 0 },
  { 0, 0, 1, - 3, 127,   4, - 1, 0 }, { 0, 0, 1, - 4, 126,   6, - 2, 1 },
  { 0, 0, 1, - 5, 126,   8, - 3, 1 }, { 0, 0, 1, - 6, 125,  11, - 4, 1 },
  { 0, 0, 1, - 7, 124,  13, - 4, 1 }, { 0, 0, 2, - 8, 123,  15, - 5, 1 },
  { 0, 0, 2, - 9, 122,  18, - 6, 1 }, { 0, 0, 2, -10, 121,  20, - 6, 1 },
  { 0, 0, 2, -11, 120,  22, - 7, 2 }, { 0, 0, 2, -12, 119,  25, - 8, 2 },
  { 0, 0, 3, -13, 117,  27, - 8, 2 }, { 0, 0, 3, -13, 116,  29, - 9, 2 },
  { 0, 0, 3, -14, 114,  32, -10, 3 }, { 0, 0, 3, -15, 113,  35, -10, 2 },
  { 0, 0, 3, -15, 111,  37, -11, 3 }, { 0, 0, 3, -16, 109,  40, -11, 3 },
  { 0, 0, 3, -16, 108,  42, -12, 3 }, { 0, 0, 4, -17, 106,  45, -13, 3 },
  { 0, 0, 4, -17, 104,  47, -13, 3 }, { 0, 0, 4, -17, 102,  50, -14, 3 },
  { 0, 0, 4, -17, 100,  52, -14, 3 }, { 0, 0, 4, -18,  98,  55, -15, 4 },
  { 0, 0, 4, -18,  96,  58, -15, 3 }, { 0, 0, 4, -18,  94,  60, -16, 4 },
  { 0, 0, 4, -18,  91,  63, -16, 4 }, { 0, 0, 4, -18,  89,  65, -16, 4 },
  { 0, 0, 4, -18,  87,  68, -17, 4 }, { 0, 0, 4, -18,  85,  70, -17, 4 },
  { 0, 0, 4, -18,  82,  73, -17, 4 }, { 0, 0, 4, -18,  80,  75, -17, 4 },
  { 0, 0, 4, -18,  78,  78, -18, 4 }, { 0, 0, 4, -17,  75,  80, -18, 4 },
  { 0, 0, 4, -17,  73,  82, -18, 4 }, { 0, 0, 4, -17,  70,  85, -18, 4 },
  { 0, 0, 4, -17,  68,  87, -18, 4 }, { 0, 0, 4, -16,  65,  89, -18, 4 },
  { 0, 0, 4, -16,  63,  91, -18, 4 }, { 0, 0, 4, -16,  60,  94, -18, 4 },
  { 0, 0, 3, -15,  58,  96, -18, 4 }, { 0, 0, 4, -15,  55,  98, -18, 4 },
  { 0, 0, 3, -14,  52, 100, -17, 4 }, { 0, 0, 3, -14,  50, 102, -17, 4 },
  { 0, 0, 3, -13,  47, 104, -17, 4 }, { 0, 0, 3, -13,  45, 106, -17, 4 },
  { 0, 0, 3, -12,  42, 108, -16, 3 }, { 0, 0, 3, -11,  40, 109, -16, 3 },
  { 0, 0, 3, -11,  37, 111, -15, 3 }, { 0, 0, 2, -10,  35, 113, -15, 3 },
  { 0, 0, 3, -10,  32, 114, -14, 3 }, { 0, 0, 2, - 9,  29, 116, -13, 3 },
  { 0, 0, 2, - 8,  27, 117, -13, 3 }, { 0, 0, 2, - 8,  25, 119, -12, 2 },
  { 0, 0, 2, - 7,  22, 120, -11, 2 }, { 0, 0, 1, - 6,  20, 121, -10, 2 },
  { 0, 0, 1, - 6,  18, 122, - 9, 2 }, { 0, 0, 1, - 5,  15, 123, - 8, 2 },
  { 0, 0, 1, - 4,  13, 124, - 7, 1 }, { 0, 0, 1, - 4,  11, 125, - 6, 1 },
  { 0, 0, 1, - 3,   8, 126, - 5, 1 }, { 0, 0, 1, - 2,   6, 126, - 4, 1 },
  { 0, 0, 0, - 1,   4, 127, - 3, 1 }, { 0, 0, 0,   0,   2, 127, - 1, 0 },
  // dummy (replicate row index 191)
  { 0, 0, 0,   0,   2, 127, - 1, 0 },
  };

static const int error_measure_lut[512] = {
  // pow 0.7
  16384, 16339, 16294, 16249, 16204, 16158, 16113, 16068,
  16022, 15977, 15932, 15886, 15840, 15795, 15749, 15703,
  15657, 15612, 15566, 15520, 15474, 15427, 15381, 15335,
  15289, 15242, 15196, 15149, 15103, 15056, 15010, 14963,
  14916, 14869, 14822, 14775, 14728, 14681, 14634, 14587,
  14539, 14492, 14445, 14397, 14350, 14302, 14254, 14206,
  14159, 14111, 14063, 14015, 13967, 13918, 13870, 13822,
  13773, 13725, 13676, 13628, 13579, 13530, 13481, 13432,
  13383, 13334, 13285, 13236, 13187, 13137, 13088, 13038,
  12988, 12939, 12889, 12839, 12789, 12739, 12689, 12639,
  12588, 12538, 12487, 12437, 12386, 12335, 12285, 12234,
  12183, 12132, 12080, 12029, 11978, 11926, 11875, 11823,
  11771, 11719, 11667, 11615, 11563, 11511, 11458, 11406,
  11353, 11301, 11248, 11195, 11142, 11089, 11036, 10982,
  10929, 10875, 10822, 10768, 10714, 10660, 10606, 10552,
  10497, 10443, 10388, 10333, 10279, 10224, 10168, 10113,
  10058, 10002,  9947,  9891,  9835,  9779,  9723,  9666,
  9610, 9553, 9497, 9440, 9383, 9326, 9268, 9211,
  9153, 9095, 9037, 8979, 8921, 8862, 8804, 8745,
  8686, 8627, 8568, 8508, 8449, 8389, 8329, 8269,
  8208, 8148, 8087, 8026, 7965, 7903, 7842, 7780,
  7718, 7656, 7593, 7531, 7468, 7405, 7341, 7278,
  7214, 7150, 7086, 7021, 6956, 6891, 6826, 6760,
  6695, 6628, 6562, 6495, 6428, 6361, 6293, 6225,
  6157, 6089, 6020, 5950, 5881, 5811, 5741, 5670,
  5599, 5527, 5456, 5383, 5311, 5237, 5164, 5090,
  5015, 4941, 4865, 4789, 4713, 4636, 4558, 4480,
  4401, 4322, 4242, 4162, 4080, 3998, 3916, 3832,
  3748, 3663, 3577, 3490, 3402, 3314, 3224, 3133,
  3041, 2948, 2854, 2758, 2661, 2562, 2461, 2359,
  2255, 2148, 2040, 1929, 1815, 1698, 1577, 1452,
  1323, 1187, 1045,  894,  731,  550,  339,    0,
  339,  550,  731,  894, 1045, 1187, 1323, 1452,
  1577, 1698, 1815, 1929, 2040, 2148, 2255, 2359,
  2461, 2562, 2661, 2758, 2854, 2948, 3041, 3133,
  3224, 3314, 3402, 3490, 3577, 3663, 3748, 3832,
  3916, 3998, 4080, 4162, 4242, 4322, 4401, 4480,
  4558, 4636, 4713, 4789, 4865, 4941, 5015, 5090,
  5164, 5237, 5311, 5383, 5456, 5527, 5599, 5670,
  5741, 5811, 5881, 5950, 6020, 6089, 6157, 6225,
  6293, 6361, 6428, 6495, 6562, 6628, 6695, 6760,
  6826, 6891, 6956, 7021, 7086, 7150, 7214, 7278,
  7341, 7405, 7468, 7531, 7593, 7656, 7718, 7780,
  7842, 7903, 7965, 8026, 8087, 8148, 8208, 8269,
  8329, 8389, 8449, 8508, 8568, 8627, 8686, 8745,
  8804, 8862, 8921, 8979, 9037, 9095, 9153, 9211,
  9268, 9326, 9383, 9440, 9497, 9553, 9610, 9666,
  9723,  9779,  9835,  9891,  9947, 10002, 10058, 10113,
  10168, 10224, 10279, 10333, 10388, 10443, 10497, 10552,
  10606, 10660, 10714, 10768, 10822, 10875, 10929, 10982,
  11036, 11089, 11142, 11195, 11248, 11301, 11353, 11406,
  11458, 11511, 11563, 11615, 11667, 11719, 11771, 11823,
  11875, 11926, 11978, 12029, 12080, 12132, 12183, 12234,
  12285, 12335, 12386, 12437, 12487, 12538, 12588, 12639,
  12689, 12739, 12789, 12839, 12889, 12939, 12988, 13038,
  13088, 13137, 13187, 13236, 13285, 13334, 13383, 13432,
  13481, 13530, 13579, 13628, 13676, 13725, 13773, 13822,
  13870, 13918, 13967, 14015, 14063, 14111, 14159, 14206,
  14254, 14302, 14350, 14397, 14445, 14492, 14539, 14587,
  14634, 14681, 14728, 14775, 14822, 14869, 14916, 14963,
  15010, 15056, 15103, 15149, 15196, 15242, 15289, 15335,
  15381, 15427, 15474, 15520, 15566, 15612, 15657, 15703,
  15749, 15795, 15840, 15886, 15932, 15977, 16022, 16068,
  16113, 16158, 16204, 16249, 16294, 16339, 16384, 16384,
};

static const uint16_t div_lut[DIV_LUT_NUM + 1] = {
  16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768,
  15709, 15650, 15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142,
  15087, 15033, 14980, 14926, 14873, 14821, 14769, 14717, 14665, 14614, 14564,
  14513, 14463, 14413, 14364, 14315, 14266, 14218, 14170, 14122, 14075, 14028,
  13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662, 13618, 13574, 13530,
  13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107, 13066,
  13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633,
  12596, 12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228,
  12193, 12157, 12122, 12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848,
  11815, 11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555, 11523, 11491,
  11460, 11429, 11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185, 11155,
  11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894, 10866, 10838,
  10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
  10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255,
  10230, 10205, 10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986,
  9963,  9939,  9916,  9892,  9869,  9846,  9823,  9800,  9777,  9754,  9732,
  9709,  9687,  9664,  9642,  9620,  9598,  9576,  9554,  9533,  9511,  9489,
  9468,  9447,  9425,  9404,  9383,  9362,  9341,  9321,  9300,  9279,  9259,
  9239,  9218,  9198,  9178,  9158,  9138,  9118,  9098,  9079,  9059,  9039,
  9020,  9001,  8981,  8962,  8943,  8924,  8905,  8886,  8867,  8849,  8830,
  8812,  8793,  8775,  8756,  8738,  8720,  8702,  8684,  8666,  8648,  8630,
  8613,  8595,  8577,  8560,  8542,  8525,  8508,  8490,  8473,  8456,  8439,
  8422,  8405,  8389,  8372,  8355,  8339,  8322,  8306,  8289,  8273,  8257,
  8240,  8224,  8208,  8192,
};

static int error_measure(int err) {
  return error_measure_lut[255 + err];
}

int64_t av1_calc_frame_error(const uint8_t *const ref, int stride,
                               const uint8_t *const dst, int p_width,
                               int p_height, int p_stride) {
  int64_t sum_error = 0;
  for (int i = 0; i < p_height; ++i) {
    for (int j = 0; j < p_width; ++j) {
      sum_error +=
          (int64_t)error_measure(dst[j + i * p_stride] - ref[j + i * stride]);
    }
  }
  return sum_error;
}

static int64_t segmented_frame_error(const uint8_t *const ref, int stride,
                                     const uint8_t *const dst, int p_width,
                                     int p_height, int p_stride,
                                     uint8_t *segment_map,
                                     int segment_map_stride) {
  int patch_w, patch_h;
  const int error_bsize_w = AOMMIN(p_width, WARP_ERROR_BLOCK);
  const int error_bsize_h = AOMMIN(p_height, WARP_ERROR_BLOCK);
  int64_t sum_error = 0;
  for (int i = 0; i < p_height; i += WARP_ERROR_BLOCK) {
    for (int j = 0; j < p_width; j += WARP_ERROR_BLOCK) {
      int seg_x = j >> WARP_ERROR_BLOCK_LOG;
      int seg_y = i >> WARP_ERROR_BLOCK_LOG;
      // Only compute the error if this block contains inliers from the motion
      // model
      if (!segment_map[seg_y * segment_map_stride + seg_x]) continue;

      // avoid computing error into the frame padding
      patch_w = AOMMIN(error_bsize_w, p_width - j);
      patch_h = AOMMIN(error_bsize_h, p_height - i);
      sum_error += av1_calc_frame_error(ref + j + i * stride, stride,
                                        dst + j + i * p_stride, patch_w,
                                        patch_h, p_stride);
    }
  }
  return sum_error;
}

#define FEAT_COUNT_TR 3
#define SEG_COUNT_TR 0.40
void av1_compute_feature_segmentation_map(uint8_t *segment_map, int width,
                                          int height, int *inliers,
                                          int num_inliers) {
  int seg_count = 0;
  memset(segment_map, 0, sizeof(*segment_map) * width * height);

  for (int i = 0; i < num_inliers; i++) {
    int x = inliers[i * 2];
    int y = inliers[i * 2 + 1];
    int seg_x = x >> WARP_ERROR_BLOCK_LOG;
    int seg_y = y >> WARP_ERROR_BLOCK_LOG;
    segment_map[seg_y * width + seg_x] += 1;
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      uint8_t feat_count = segment_map[i * width + j];
      segment_map[i * width + j] = (feat_count >= FEAT_COUNT_TR);
      seg_count += (segment_map[i * width + j]);
    }
  }

  // If this motion does not make up a large enough portion of the frame,
  // use the unsegmented version of the error metric
  if (seg_count < (width * height * SEG_COUNT_TR))
    memset(segment_map, 1, width * height * sizeof(*segment_map));
}

int64_t av1_segmented_frame_error(const uint8_t *ref,
                                  int stride, uint8_t *dst, int p_width,
                                  int p_height, int p_stride,
                                  uint8_t *segment_map,
                                  int segment_map_stride) {
  return segmented_frame_error(ref, stride, dst, p_width, p_height, p_stride,
                               segment_map, segment_map_stride);
}

static const double erroradv_tr = 0.65;
static int64_t erroradv_threshold(int64_t ref_frame_error) {
  return (int64_t)(ref_frame_error * erroradv_tr + 0.5);
}

static enum TransformationType get_wmtype(const WarpedMotionParams *gm) {
  if (gm->wmmat[5] == (1 << WARPEDMODEL_PREC_BITS) && !gm->wmmat[4] &&
      gm->wmmat[2] == (1 << WARPEDMODEL_PREC_BITS) && !gm->wmmat[3]) {
    return ((!gm->wmmat[1] && !gm->wmmat[0]) ? IDENTITY : TRANSLATION);
  }
  if (gm->wmmat[2] == gm->wmmat[5] && gm->wmmat[3] == -gm->wmmat[4])
    return ROTZOOM;
  else
    return AFFINE;
}

static void force_wmtype(WarpedMotionParams *wm, enum TransformationType wmtype) {
  switch (wmtype) {
    case IDENTITY:
      wm->wmmat[0] = 0;
      wm->wmmat[1] = 0;
      return;
    case TRANSLATION:
      wm->wmmat[2] = 1 << WARPEDMODEL_PREC_BITS;
      wm->wmmat[3] = 0;
      return;
    case ROTZOOM:
      wm->wmmat[4] = -wm->wmmat[3];
      wm->wmmat[5] = wm->wmmat[2];
      return;
    case AFFINE: break;
    default: return;
  }
  wm->wmtype = wmtype;
}

static int is_affine_valid(const WarpedMotionParams *const wm) {
  const int32_t *mat = wm->wmmat;
  return (mat[2] > 0);
}

static int clamp(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

static int get_msb(unsigned int n) {
  int log = 0;
  unsigned int value = n;
  int i;

  assert(n != 0);

  for (i = 4; i >= 0; --i) {
    const int shift = (1 << i);
    const unsigned int x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}

static int16_t resolve_divisor_32(uint32_t D, int16_t *shift) {
  int32_t f;
  *shift = get_msb(D);
  // e is obtained from D after resetting the most significant 1 bit.
  const int32_t e = D - ((uint32_t)1 << *shift);
  // Get the most significant DIV_LUT_BITS (8) bits of e into f
  if (*shift > DIV_LUT_BITS)
    f = ROUND_POWER_OF_TWO(e, *shift - DIV_LUT_BITS);
  else
    f = e << (DIV_LUT_BITS - *shift);
  assert(f <= DIV_LUT_NUM);
  *shift += DIV_LUT_PREC_BITS;
  // Use f as lookup into the precomputed table of multipliers
  return div_lut[f];
}

static int is_affine_shear_allowed(int16_t alpha, int16_t beta, int16_t gamma,
                                   int16_t delta) {
  if ((4 * abs(alpha) + 7 * abs(beta) >= (1 << WARPEDMODEL_PREC_BITS)) ||
      (4 * abs(gamma) + 4 * abs(delta) >= (1 << WARPEDMODEL_PREC_BITS)))
    return 0;
  else
    return 1;
}

int av1_get_shear_params(WarpedMotionParams *wm) {
  const int32_t *mat = wm->wmmat;
  if (!is_affine_valid(wm)) return 0;
  wm->alpha =
      clamp(mat[2] - (1 << WARPEDMODEL_PREC_BITS), INT16_MIN, INT16_MAX);
  wm->beta = clamp(mat[3], INT16_MIN, INT16_MAX);
  int16_t shift;
  int16_t y = resolve_divisor_32(abs(mat[2]), &shift) * (mat[2] < 0 ? -1 : 1);
  int64_t v = ((int64_t)mat[4] * (1 << WARPEDMODEL_PREC_BITS)) * y;
  wm->gamma =
      clamp((int)ROUND_POWER_OF_TWO_SIGNED_64(v, shift), INT16_MIN, INT16_MAX);
  v = ((int64_t)mat[3] * mat[4]) * y;
  wm->delta = clamp(mat[5] - (int)ROUND_POWER_OF_TWO_SIGNED_64(v, shift) -
                        (1 << WARPEDMODEL_PREC_BITS),
                    INT16_MIN, INT16_MAX);

  wm->alpha = ROUND_POWER_OF_TWO_SIGNED(wm->alpha, WARP_PARAM_REDUCE_BITS) *
              (1 << WARP_PARAM_REDUCE_BITS);
  wm->beta = ROUND_POWER_OF_TWO_SIGNED(wm->beta, WARP_PARAM_REDUCE_BITS) *
             (1 << WARP_PARAM_REDUCE_BITS);
  wm->gamma = ROUND_POWER_OF_TWO_SIGNED(wm->gamma, WARP_PARAM_REDUCE_BITS) *
              (1 << WARP_PARAM_REDUCE_BITS);
  wm->delta = ROUND_POWER_OF_TWO_SIGNED(wm->delta, WARP_PARAM_REDUCE_BITS) *
              (1 << WARP_PARAM_REDUCE_BITS);

  if (!is_affine_shear_allowed(wm->alpha, wm->beta, wm->gamma, wm->delta))
    return 0;

  return 1;
}

static uint8_t clip_pixel(int val) {
  return (val > 255) ? 255 : (val < 0) ? 0 : val;
}

void av1_warp_affine(const int32_t *mat, const uint8_t *ref, int width,
                       int height, int stride, uint8_t *pred, int p_col,
                       int p_row, int p_width, int p_height, int p_stride,
                       int subsampling_x, int subsampling_y,
                       ConvolveParams *conv_params, int16_t alpha, int16_t beta,
                       int16_t gamma, int16_t delta) {
  int32_t tmp[15 * 8];
  const int bd = 8;
  const int reduce_bits_horiz = conv_params->round_0;
  const int reduce_bits_vert = conv_params->is_compound
                                   ? conv_params->round_1
                                   : 2 * FILTER_BITS - reduce_bits_horiz;
  const int max_bits_horiz = bd + FILTER_BITS + 1 - reduce_bits_horiz;
  const int offset_bits_horiz = bd + FILTER_BITS - 1;
  const int offset_bits_vert = bd + 2 * FILTER_BITS - reduce_bits_horiz;
  const int round_bits =
      2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
  const int offset_bits = bd + 2 * FILTER_BITS - conv_params->round_0;
  (void)max_bits_horiz;
  assert(IMPLIES(conv_params->is_compound, conv_params->dst != NULL));
  assert(IMPLIES(conv_params->do_average, conv_params->is_compound));

  for (int i = p_row; i < p_row + p_height; i += 8) {
    for (int j = p_col; j < p_col + p_width; j += 8) {
      // Calculate the center of this 8x8 block,
      // project to luma coordinates (if in a subsampled chroma plane),
      // apply the affine transformation,
      // then convert back to the original coordinates (if necessary)
      const int32_t src_x = (j + 4) << subsampling_x;
      const int32_t src_y = (i + 4) << subsampling_y;
      const int64_t dst_x =
          (int64_t)mat[2] * src_x + (int64_t)mat[3] * src_y + (int64_t)mat[0];
      const int64_t dst_y =
          (int64_t)mat[4] * src_x + (int64_t)mat[5] * src_y + (int64_t)mat[1];
      const int64_t x4 = dst_x >> subsampling_x;
      const int64_t y4 = dst_y >> subsampling_y;

      int32_t ix4 = (int32_t)(x4 >> WARPEDMODEL_PREC_BITS);
      int32_t sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
      int32_t iy4 = (int32_t)(y4 >> WARPEDMODEL_PREC_BITS);
      int32_t sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);

      sx4 += alpha * (-4) + beta * (-4);
      sy4 += gamma * (-4) + delta * (-4);

      sx4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);
      sy4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);

      // Horizontal filter
      for (int k = -7; k < 8; ++k) {
        // Clamp to top/bottom edge of the frame
        const int iy = clamp(iy4 + k, 0, height - 1);

        int sx = sx4 + beta * (k + 4);

        for (int l = -4; l < 4; ++l) {
          int ix = ix4 + l - 3;
          // At this point, sx = sx4 + alpha * l + beta * k
          const int offs = ROUND_POWER_OF_TWO(sx, WARPEDDIFF_PREC_BITS) +
                           WARPEDPIXEL_PREC_SHIFTS;
          assert(offs >= 0 && offs <= WARPEDPIXEL_PREC_SHIFTS * 3);
          const int16_t *coeffs = av1_warped_filter[offs];

          int32_t sum = 1 << offset_bits_horiz;
          for (int m = 0; m < 8; ++m) {
            // Clamp to left/right edge of the frame
            const int sample_x = clamp(ix + m, 0, width - 1);

            sum += ref[iy * stride + sample_x] * coeffs[m];
          }
          sum = ROUND_POWER_OF_TWO(sum, reduce_bits_horiz);
          assert(0 <= sum && sum < (1 << max_bits_horiz));
          tmp[(k + 7) * 8 + (l + 4)] = sum;
          sx += alpha;
        }
      }

      // Vertical filter
      for (int k = -4; k < AOMMIN(4, p_row + p_height - i - 4); ++k) {
        int sy = sy4 + delta * (k + 4);
        for (int l = -4; l < AOMMIN(4, p_col + p_width - j - 4); ++l) {
          // At this point, sy = sy4 + gamma * l + delta * k
          const int offs = ROUND_POWER_OF_TWO(sy, WARPEDDIFF_PREC_BITS) +
                           WARPEDPIXEL_PREC_SHIFTS;
          assert(offs >= 0 && offs <= WARPEDPIXEL_PREC_SHIFTS * 3);
          const int16_t *coeffs = av1_warped_filter[offs];

          int32_t sum = 1 << offset_bits_vert;
          for (int m = 0; m < 8; ++m) {
            sum += tmp[(k + m + 4) * 8 + (l + 4)] * coeffs[m];
          }

          if (conv_params->is_compound) {
            CONV_BUF_TYPE *p =
                &conv_params
                     ->dst[(i - p_row + k + 4) * conv_params->dst_stride +
                           (j - p_col + l + 4)];
            sum = ROUND_POWER_OF_TWO(sum, reduce_bits_vert);
            if (conv_params->do_average) {
              uint8_t *dst8 =
                  &pred[(i - p_row + k + 4) * p_stride + (j - p_col + l + 4)];
              int32_t tmp32 = *p;
              if (conv_params->use_dist_wtd_comp_avg) {
                tmp32 = tmp32 * conv_params->fwd_offset +
                        sum * conv_params->bck_offset;
                tmp32 = tmp32 >> DIST_PRECISION_BITS;
              } else {
                tmp32 += sum;
                tmp32 = tmp32 >> 1;
              }
              tmp32 = tmp32 - (1 << (offset_bits - conv_params->round_1)) -
                      (1 << (offset_bits - conv_params->round_1 - 1));
              *dst8 = clip_pixel(ROUND_POWER_OF_TWO(tmp32, round_bits));
            } else {
              *p = sum;
            }
          } else {
            uint8_t *p =
                &pred[(i - p_row + k + 4) * p_stride + (j - p_col + l + 4)];
            sum = ROUND_POWER_OF_TWO(sum, reduce_bits_vert);
            assert(0 <= sum && sum < (1 << (bd + 2)));
            *p = clip_pixel(sum - (1 << (bd - 1)) - (1 << bd));
          }
          sy += gamma;
        }
      }
    }
  }
}

void warp_plane(WarpedMotionParams *wm, const uint8_t *const ref, int width,
                int height, int stride, uint8_t *pred, int p_col, int p_row,
                int p_width, int p_height, int p_stride, int subsampling_x,
                int subsampling_y, ConvolveParams *conv_params) {
  assert(wm->wmtype <= AFFINE);
  if (wm->wmtype == ROTZOOM) {
    wm->wmmat[5] = wm->wmmat[2];
    wm->wmmat[4] = -wm->wmmat[3];
  }
  const int32_t *const mat = wm->wmmat;
  const int16_t alpha = wm->alpha;
  const int16_t beta = wm->beta;
  const int16_t gamma = wm->gamma;
  const int16_t delta = wm->delta;
  av1_warp_affine(mat, ref, width, height, stride, pred, p_col, p_row, p_width,
                  p_height, p_stride, subsampling_x, subsampling_y, conv_params,
                  alpha, beta, gamma, delta);
}

static ConvolveParams get_conv_params_no_round(int cmp_index, int plane,
                                                      CONV_BUF_TYPE *dst,
                                                      int dst_stride,
                                                      int is_compound, int bd) {
  ConvolveParams conv_params;
  assert(IMPLIES(cmp_index, is_compound));

  conv_params.is_compound = is_compound;
  conv_params.use_dist_wtd_comp_avg = 0;
  conv_params.round_0 = ROUND0_BITS;
  conv_params.round_1 = is_compound ? COMPOUND_ROUND1_BITS
                                    : 2 * FILTER_BITS - conv_params.round_0;
  const int intbufrange = bd + FILTER_BITS - conv_params.round_0 + 2;
  assert(IMPLIES(bd < 12, intbufrange <= 16));
  if (intbufrange > 16) {
    conv_params.round_0 += intbufrange - 16;
    if (!is_compound) conv_params.round_1 -= intbufrange - 16;
  }
  // TODO(yunqing): The following dst should only be valid while
  // is_compound = 1;
  conv_params.dst = dst;
  conv_params.dst_stride = dst_stride;
  conv_params.plane = plane;

  // By default, set do average to 1 if this is the second single prediction
  // in a compound mode.
  conv_params.do_average = cmp_index;
  return conv_params;
}

static ConvolveParams get_conv_params(int do_average, int plane,
                                             int bd) {
  return get_conv_params_no_round(do_average, plane, NULL, 0, 0, bd);
}


static int64_t warp_error(WarpedMotionParams *wm, const uint8_t *const ref,
                          int width, int height, int stride,
                          const uint8_t *const dst, int p_col, int p_row,
                          int p_width, int p_height, int p_stride,
                          int subsampling_x, int subsampling_y,
                          int64_t best_error, uint8_t *segment_map,
                          int segment_map_stride) {
  int64_t gm_sumerr = 0;
  int warp_w, warp_h;
  const int error_bsize_w = AOMMIN(p_width, WARP_ERROR_BLOCK);
  const int error_bsize_h = AOMMIN(p_height, WARP_ERROR_BLOCK);
  DECLARE_ALIGNED(16, uint8_t, tmp[WARP_ERROR_BLOCK * WARP_ERROR_BLOCK]);
  ConvolveParams conv_params = get_conv_params(0, 0, 8);
  conv_params.use_dist_wtd_comp_avg = 0;

  for (int i = p_row; i < p_row + p_height; i += WARP_ERROR_BLOCK) {
    for (int j = p_col; j < p_col + p_width; j += WARP_ERROR_BLOCK) {
      int seg_x = j >> WARP_ERROR_BLOCK_LOG;
      int seg_y = i >> WARP_ERROR_BLOCK_LOG;
      // Only compute the error if this block contains inliers from the motion
      // model
      if (!segment_map[seg_y * segment_map_stride + seg_x]) continue;
      // avoid warping extra 8x8 blocks in the padded region of the frame
      // when p_width and p_height are not multiples of WARP_ERROR_BLOCK
      warp_w = AOMMIN(error_bsize_w, p_col + p_width - j);
      warp_h = AOMMIN(error_bsize_h, p_row + p_height - i);
      warp_plane(wm, ref, width, height, stride, tmp, j, i, warp_w, warp_h,
                 WARP_ERROR_BLOCK, subsampling_x, subsampling_y, &conv_params);

      gm_sumerr +=
          av1_calc_frame_error(tmp, WARP_ERROR_BLOCK, dst + j + i * p_stride,
                               warp_w, warp_h, p_stride);
      if (gm_sumerr > best_error) return INT64_MAX;
    }
  }
  return gm_sumerr;
}

int64_t av1_warp_error(WarpedMotionParams *wm, int use_hbd, int bd,
                       const uint8_t *ref, int width, int height, int stride,
                       uint8_t *dst, int p_col, int p_row, int p_width,
                       int p_height, int p_stride, int subsampling_x,
                       int subsampling_y, int64_t best_error,
                       uint8_t *segment_map, int segment_map_stride) {
  if (wm->wmtype <= AFFINE)
    if (!av1_get_shear_params(wm)) return INT64_MAX;

  return warp_error(wm, ref, width, height, stride, dst, p_col, p_row, p_width,
                    p_height, p_stride, subsampling_x, subsampling_y,
                    best_error, segment_map, segment_map_stride);
}

static int64_t calc_approx_erroradv_threshold(
    double scaling_factor, int64_t erroradv_threshold) {
  return erroradv_threshold <
                 (int64_t)(((double)INT64_MAX / scaling_factor) + 0.5)
             ? (int64_t)(scaling_factor * erroradv_threshold + 0.5)
             : INT64_MAX;
}

static double thresh_factors[GM_REFINEMENT_COUNT] = { 1.25, 1.20, 1.15, 1.10,
                                                      1.05 };

static int32_t add_param_offset(int param_index, int32_t param_value,
                                int32_t offset) {
  const int scale_vals[3] = { GM_TRANS_PREC_DIFF, GM_ALPHA_PREC_DIFF,
                              GM_ROW3HOMO_PREC_DIFF };
  const int clamp_vals[3] = { GM_TRANS_MAX, GM_ALPHA_MAX, GM_ROW3HOMO_MAX };
  // type of param: 0 - translation, 1 - affine, 2 - homography
  const int param_type = (param_index < 2 ? 0 : (param_index < 6 ? 1 : 2));
  const int is_one_centered = (param_index == 2 || param_index == 5);

  // Make parameter zero-centered and offset the shift that was done to make
  // it compatible with the warped model
  param_value = (param_value - (is_one_centered << WARPEDMODEL_PREC_BITS)) >>
                scale_vals[param_type];
  // Add desired offset to the rescaled/zero-centered parameter
  param_value += offset;
  // Clamp the parameter so it does not overflow the number of bits allotted
  // to it in the bitstream
  param_value = (int32_t)clamp(param_value, -clamp_vals[param_type],
                               clamp_vals[param_type]);
  // Rescale the parameter to WARPEDMODEL_PRECISION_BITS so it is compatible
  // with the warped motion library
  param_value *= (1 << scale_vals[param_type]);

  // Undo the zero-centering step if necessary
  return param_value + (is_one_centered << WARPEDMODEL_PREC_BITS);
}

int64_t av1_refine_integerized_param(
    WarpedMotionParams *wm, enum TransformationType wmtype, int use_hbd, int bd,
    uint8_t *ref, int r_width, int r_height, int r_stride, uint8_t *dst,
    int d_width, int d_height, int d_stride, int n_refinements,
    int64_t best_frame_error, uint8_t *segment_map, int segment_map_stride,
    int64_t erroradv_threshold) {
  static const int max_trans_model_params[TRANS_TYPES] = { 0, 2, 4, 6 };
  const int border = ERRORADV_BORDER;
  int i = 0, p;
  int n_params = max_trans_model_params[wmtype];
  int32_t *param_mat = wm->wmmat;
  int64_t step_error, best_error;
  int32_t step;
  int32_t *param;
  int32_t curr_param;
  int32_t best_param;

  force_wmtype(wm, wmtype);
  best_error =
      av1_warp_error(wm, use_hbd, bd, ref, r_width, r_height, r_stride,
                     dst + border * d_stride + border, border, border,
                     d_width - 2 * border, d_height - 2 * border, d_stride, 0,
                     0, best_frame_error, segment_map, segment_map_stride);
  best_error = AOMMIN(best_error, best_frame_error);
  step = 1 << (n_refinements - 1);
  for (i = 0; i < n_refinements; i++, step >>= 1) {
    int64_t error_adv_thresh =
        calc_approx_erroradv_threshold(thresh_factors[i], erroradv_threshold);
    for (p = 0; p < n_params; ++p) {
      int step_dir = 0;
      // Skip searches for parameters that are forced to be 0
      param = param_mat + p;
      curr_param = *param;
      best_param = curr_param;
      // look to the left
      *param = add_param_offset(p, curr_param, -step);
      step_error =
          av1_warp_error(wm, use_hbd, bd, ref, r_width, r_height, r_stride,
                         dst + border * d_stride + border, border, border,
                         d_width - 2 * border, d_height - 2 * border, d_stride,
                         0, 0, AOMMIN(best_error, error_adv_thresh),
                         segment_map, segment_map_stride);
      if (step_error < best_error) {
        best_error = step_error;
        best_param = *param;
        step_dir = -1;
      }

      // look to the right
      *param = add_param_offset(p, curr_param, step);
      step_error =
          av1_warp_error(wm, use_hbd, bd, ref, r_width, r_height, r_stride,
                         dst + border * d_stride + border, border, border,
                         d_width - 2 * border, d_height - 2 * border, d_stride,
                         0, 0, AOMMIN(best_error, error_adv_thresh),
                         segment_map, segment_map_stride);
      if (step_error < best_error) {
        best_error = step_error;
        best_param = *param;
        step_dir = 1;
      }
      *param = best_param;

      // look to the direction chosen above repeatedly until error increases
      // for the biggest step size
      while (step_dir) {
        *param = add_param_offset(p, best_param, step * step_dir);
        step_error =
            av1_warp_error(wm, use_hbd, bd, ref, r_width, r_height, r_stride,
                           dst + border * d_stride + border, border, border,
                           d_width - 2 * border, d_height - 2 * border,
                           d_stride, 0, 0, AOMMIN(best_error, error_adv_thresh),
                           segment_map, segment_map_stride);
        if (step_error < best_error) {
          best_error = step_error;
          best_param = *param;
        } else {
          *param = best_param;
          step_dir = 0;
        }
      }
    }
  }
  force_wmtype(wm, wmtype);
  wm->wmtype = get_wmtype(wm);
  return best_error;
}