#ifndef __AFEST__
#define __AFEST__

#include "Base.cpp"

#endif  // End of '__AFEST__'

class AffineEstimator
    : virtual public RTL::Estimator<RTL::Affine, Correspondence,
                                    Correspondence *> {
 public:
  RTL::Affine computeModel(double *points1, double *points2, int np) override {
    assert(np > 0);
    const int np2 = np * 2;

    double *a = (double *)malloc(sizeof(*a) * (np2 * 7 + 42));

    double *b = a + np2 * 6;
    double *temp = b + np2;
    double *pts1 = points1;
    double *pts2 = points2;

    double sx, sy, dx, dy;
    double T1[9], T2[9];
    normalize_homography(pts1, np, T1);
    normalize_homography(pts2, np, T2);

    for (int i = 0; i < np; ++i) {
      dx = *(pts2++);
      dy = *(pts2++);
      sx = *(pts1++);
      sy = *(pts1++);

      a[i * 2 * 6 + 0] = sx;
      a[i * 2 * 6 + 1] = sy;
      a[i * 2 * 6 + 2] = 0;
      a[i * 2 * 6 + 3] = 0;
      a[i * 2 * 6 + 4] = 1;
      a[i * 2 * 6 + 5] = 0;
      a[(i * 2 + 1) * 6 + 0] = 0;
      a[(i * 2 + 1) * 6 + 1] = 0;
      a[(i * 2 + 1) * 6 + 2] = sx;
      a[(i * 2 + 1) * 6 + 3] = sy;
      a[(i * 2 + 1) * 6 + 4] = 0;
      a[(i * 2 + 1) * 6 + 5] = 1;

      b[2 * i] = dx;
      b[2 * i + 1] = dy;
    }

    RTL::Affine affine;

    if (!least_squares(6, a, np2, 6, b, temp, affine.mat)) {
      free(a);
      affine.lse = false;
      return affine;
    }

    denormalize_affine_reorder(affine.mat, T1, T2);
    free(a);
    affine.lse = true;
    return affine;
  }

  double computeError(const RTL::Affine &affine,
                      const Correspondence &correspondence) override {
    double sx = correspondence.x;
    double sy = correspondence.y;
    double rx = correspondence.rx;
    double ry = correspondence.ry;
    const double *mat = affine.mat;

    double px = mat[2] * sx + mat[3] * sy + mat[0];
    double py = mat[4] * sx + mat[5] * sy + mat[1];

    double dx = px - rx;
    double dy = py - ry;
    double distance = sqrt(dx * dx + dy * dy);

    return distance;
  }

  bool fitsLSE(const RTL::Affine &affine) override { return affine.lse; }

  bool isDegenerate(double *p) override {
    return is_collinear3(p, p + 2, p + 4);
  }

  static int is_collinear3(const double *p1, const double *p2,
                           const double *p3) {
    static const double collinear_eps = 1e-3;
    const double v =
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
    return fabs(v) < collinear_eps;
  }

  static void denormalize_affine_reorder(double *params, double *T1,
                                         double *T2) {
    double params_denorm[MAX_PARAMDIM];
    params_denorm[0] = params[0];
    params_denorm[1] = params[1];
    params_denorm[2] = params[4];
    params_denorm[3] = params[2];
    params_denorm[4] = params[3];
    params_denorm[5] = params[5];
    params_denorm[6] = params_denorm[7] = 0;
    params_denorm[8] = 1;
    denormalize_homography(params_denorm, T1, T2);
    params[0] = params_denorm[2];
    params[1] = params_denorm[5];
    params[2] = params_denorm[0];
    params[3] = params_denorm[1];
    params[4] = params_denorm[3];
    params[5] = params_denorm[4];
    params[6] = params[7] = 0;
  }
};
