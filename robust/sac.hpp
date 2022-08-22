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

#ifdef __cplusplus
extern "C" {
#endif

void estimate(Correspondence *correspondences, int num_correspondences,
              TransformationType transformation_type, Estimate type, Stats &stats);

void estimate_av1(int *correspondences, int num_correspondences,
              TransformationType transformation_type, Stats &stats);

#ifdef __cplusplus
}
#endif

#endif  // TCC_SAC_HPP
