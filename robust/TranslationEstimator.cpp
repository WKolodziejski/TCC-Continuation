#ifndef __TREST__
#define __TREST__

#include "Base.cpp"

#endif  // End of '__TREST__'

class TranslationEstimator
    : virtual public RTL::Estimator<RTL::Affine, Correspondence,
                                    Correspondence *> {
 public:
  RTL::Affine computeModel(double *points1, double *points2, int np) override {
    assert(np > 0);
    double *pts1 = points1;
    double *pts2 = points2;

    double sx, sy, dx, dy;
    double sumx, sumy;
    double T1[9], T2[9];
    normalize_homography(pts1, np, T1);
    normalize_homography(pts2, np, T2);

    sumx = 0;
    sumy = 0;
    for (int i = 0; i < np; ++i) {
      dx = *(pts2++);
      dy = *(pts2++);
      sx = *(pts1++);
      sy = *(pts1++);

      sumx += dx - sx;
      sumy += dy - sy;
    }

    RTL::Affine affine;

    affine.mat[0] = sumx / np;
    affine.mat[1] = sumy / np;
    affine.lse = true;
    denormalize_translation_reorder(affine.mat, T1, T2);

    return affine;
  }

  double computeError(const RTL::Affine &affine,
                      const Correspondence &correspondence) override {
    double sx = correspondence.x;
    double sy = correspondence.y;
    double rx = correspondence.rx;
    double ry = correspondence.ry;
    const double *mat = affine.mat;

    double px = sx + mat[0];
    double py = sy + mat[1];

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

  static void denormalize_translation_reorder(double *params, double *T1,
                                              double *T2) {
    double params_denorm[MAX_PARAMDIM];
    params_denorm[0] = 1;
    params_denorm[1] = 0;
    params_denorm[2] = params[0];
    params_denorm[3] = 0;
    params_denorm[4] = 1;
    params_denorm[5] = params[1];
    params_denorm[6] = params_denorm[7] = 0;
    params_denorm[8] = 1;
    denormalize_homography(params_denorm, T1, T2);
    params[0] = params_denorm[2];
    params[1] = params_denorm[5];
    params[2] = params[5] = 1;
    params[3] = params[4] = 0;
    params[6] = params[7] = 0;
  }
};