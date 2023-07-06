#ifndef WILLIAM_H
#define WILLIAM_H

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "av1/corner_match.h"

#ifdef __cplusplus
extern "C" {
#endif

using namespace std;

#define WARP_BLOCK_SIZE 32

struct MatrixMap {
  double mat[8]{};
  double error = HUGE_VAL;
  int k;
};

struct Stats {
  int src_points_num = 0;
  int ref_points_num = 0;
  int matches_num = 0;
  int inliers_num = 0;
  int outliers_num = 0;
  float inliers_per = 0;
  unsigned long time = 0;
};

enum Detect {
  FAST_BEBLID,
  FAST_SIFT,
  FAST_SURF,
  FAST_BRISK,
  FAST_BRIEF,
  FAST_FREAK,
  FAST_DAISY,
  //  AGAST_BEBLID,
  //  AGAST_SIFT,
  //  AGAST_SURF,
  //  AGAST_BRISK,
  //  AGAST_BRIEF,
  //  AGAST_FREAK,
  //  AGAST_DAISY,
  SIFT,
  SURF,
  ORB,
  BRISK,
  KAZE,
  AKAZE,
  FAST_AOM
};

enum Match { FLANN_BEST, FLANN_KNN, BF_BEST, BF_KNN, BF_AOM };

enum Estimate {
  RANSAC,
  MLESAC,
  MSAC,
  PROSAC,
  PROMSAC,
  PROMLESAC,
  LMEDS,
  RANSAC_AOM
};

void compute(const cv::Mat &src_frame, const cv::Mat &ref_frame,
             TransformationType transformation_type, Detect detect_type,
             Match match_type, Estimate estimate_type, Stats &stats, int frame);

void av1(const cv::Mat &src_frame, const cv::Mat &ref_frame,
         TransformationType transformation_type, Stats &stats, int frame);

string formatNameSolution(Detect detect_type, Match match_type, Estimate estimate_type,
                  int frame);

string detectName(int d);

string matchName(int m);

string describeName(int d);

string modeName(int m);

string estimateName(int e);

#ifdef __cplusplus
}
#endif

#endif  // WILLIAM_H
