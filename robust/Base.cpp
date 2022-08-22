#ifndef __RTL_BASE__
#define __RTL_BASE__

#define MAX_MINPTS 4
#define MAX_DEGENERATE_ITER 10
#define MINPTS_MULTIPLIER 5

#define INLIER_THRESHOLD 1.25
#define MIN_TRIALS 2000
//#define MAX_TRIALS 10

#define MAX_PARAMDIM 9

#define MIN_PTS 3

#define SIGMA_SCALE 1.96
#define ITERATION 5

#define MIN_INLIER_PROB 0.1

#include <set>
#include <vector>
#include "../av1/mathutils.h"
#include "../av1/corner_match.h"
#include <cassert>

namespace RTL {

class Affine {
 public:
  Affine() = default;

  double mat[9]{ 0 };
  bool lse{ false };
};

template <class Model, class Datum, class Data>
class Estimator {
 public:
  virtual Model computeModel(double *points1, double *points2, int np) = 0;

  virtual double computeError(const Model &model, const Datum &datum) = 0;

  virtual bool fitsLSE(const Model &model) = 0;

  virtual bool isDegenerate(double *p) = 0;

  virtual ~Estimator() = default;

 protected:
  static void normalize_homography(double *pts, int n, double *T) {
    double *p = pts;
    double mean[2] = { 0, 0 };
    double msqe = 0;
    double scale;
    int i;

    assert(n > 0);
    for (i = 0; i < n; ++i, p += 2) {
      mean[0] += p[0];
      mean[1] += p[1];
    }
    mean[0] /= n;
    mean[1] /= n;
    for (p = pts, i = 0; i < n; ++i, p += 2) {
      p[0] -= mean[0];
      p[1] -= mean[1];
      msqe += sqrt(p[0] * p[0] + p[1] * p[1]);
    }
    msqe /= n;
    scale = (msqe == 0 ? 1.0 : sqrt(2) / msqe);
    T[0] = scale;
    T[1] = 0;
    T[2] = -scale * mean[0];
    T[3] = 0;
    T[4] = scale;
    T[5] = -scale * mean[1];
    T[6] = 0;
    T[7] = 0;
    T[8] = 1;
    for (p = pts, i = 0; i < n; ++i, p += 2) {
      p[0] *= scale;
      p[1] *= scale;
    }
  }

  static void denormalize_homography(double *params, double *T1, double *T2) {
    double iT2[9];
    double params2[9];
    invnormalize_mat(T2, iT2);
    multiply_mat(params, T1, params2, 3, 3, 3);
    multiply_mat(iT2, params2, params, 3, 3, 3);
  }

  static void invnormalize_mat(const double *T, double *iT) {
    double is = 1.0 / T[0];
    double m0 = -T[2] * is;
    double m1 = -T[5] * is;
    iT[0] = is;
    iT[1] = 0;
    iT[2] = m0;
    iT[3] = 0;
    iT[4] = is;
    iT[5] = m1;
    iT[6] = 0;
    iT[7] = 0;
    iT[8] = 1;
  }
};

}  // namespace RTL

#endif  // End of '__RTL_BASE__'
