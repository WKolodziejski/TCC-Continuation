#ifndef TCC_SAC_HPP
#define TCC_SAC_HPP

#include "RANSAC.cpp"
#include "LMedS.cpp"
#include "MSAC.cpp"
#include "MLESAC.cpp"
#include "PROSAC.cpp"
#include "PROMSAC.cpp"
#include "PROMLESAC.cpp"
#include "AffineEstimator.cpp"
#include "RotZoomEstimator.cpp"
#include "TranslationEstimator.cpp"
#include <cstdio>
#include "../av1/corner_match.h"
#include "../william.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

double estimate(Correspondence *correspondences, int num_correspondences,
              TransformationType transformation_type, Estimate type,
              Stats &stats, double mat[8]);

void estimate_av1(int *correspondences, int num_correspondences,
                  TransformationType transformation_type, Stats &stats);

void estimate_clustered(Mat &src_img, Mat &ref_img,
                        Correspondence *correspondences,
                        int num_correspondences,
                        TransformationType transformation_type, Estimate type,
                        Stats &stats, int frame);

#ifdef __cplusplus
}
#endif

#endif  // TCC_SAC_HPP
