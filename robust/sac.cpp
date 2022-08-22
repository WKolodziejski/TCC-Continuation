#ifndef __SAC__
#define __SAC__

#include "sac.hpp"
#include "../av1/ransac.c"

RTL::Estimator<RTL::Affine, Correspondence, Correspondence *> *getEstimator(
    TransformationType type) {
  switch (type) {
    case AFFINE: return new AffineEstimator();
    case ROTZOOM: return new RotZoomEstimator();
    case TRANSLATION: return new TranslationEstimator();
    default: assert(0);
  }
}

void estimate(Correspondence *correspondences, int num_correspondences,
              TransformationType transformation_type, Estimate type, Stats &stats) {
  RTL::Estimator<RTL::Affine, Correspondence, Correspondence *> *estimator =
      getEstimator(transformation_type);
  MotionModel params_by_motion[1];
  params_by_motion[0].num_inliers = 0;
  params_by_motion[0].inliers = static_cast<int *>(
      malloc(sizeof(*(params_by_motion[0].inliers)) * 2 * num_correspondences));

  if (type == Estimate::RANSAC) {
    RTL::RANSAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::MLESAC) {
    RTL::MLESAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::MSAC) {
    RTL::MSAC<RTL::Affine, Correspondence, Correspondence *> ransac(estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::PROSAC) {
    RTL::PROSAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::LMEDS) {
    RTL::LMedS<RTL::Affine, Correspondence, Correspondence *> ransac(estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::PROMSAC) {
    RTL::PROMSAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::PROMLESAC) {
    RTL::PROMLESAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else {
    assert(0);
  }

  stats.outliers_num = num_correspondences - params_by_motion->num_inliers;
  stats.inliers_num = params_by_motion->num_inliers;

  float inliers_per =
      stats.matches_num == 0
          ? 0.0
          : (float)stats.inliers_num / (float)stats.matches_num * 100;

  stats.inliers_per = inliers_per;

  // Pontos: inliers / outliers
//  fprintf(stderr, "%d / %d : %f\n", params_by_motion->num_inliers, outliers, p);

  free(params_by_motion[0].inliers);

  delete estimator;
}

void estimate_av1(int *correspondences, int num_correspondences,
                  TransformationType transformation_type, Stats &stats) {
  int num_inliers_by_motion[1];
  MotionModel params_by_motion[1];
  params_by_motion[0].num_inliers = 0;
  params_by_motion[0].inliers = static_cast<int *>(
      malloc(sizeof(*(params_by_motion[0].inliers)) * 2 * num_correspondences));

  RansacFunc ransac = av1_get_ransac_type(transformation_type);
  ransac(correspondences, num_correspondences, num_inliers_by_motion,
         params_by_motion, 1);

  stats.outliers_num = num_correspondences - num_inliers_by_motion[0];
  stats.inliers_num = num_inliers_by_motion[0];

  float inliers_per =
      stats.matches_num == 0
          ? 0.0
          : (float)stats.inliers_num / (float)stats.matches_num * 100;

  stats.inliers_per = inliers_per;

  // Pontos: inliers / outliers
  //  fprintf(stderr, "%d / %d : %f\n", params_by_motion->num_inliers, outliers,
  //  p);

  free(params_by_motion[0].inliers);
}

#endif  // End of '__SAC__'