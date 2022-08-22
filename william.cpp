#include "william.hpp"
#include "robust/sac.hpp"
#include "opencv/cv.cpp"
#include "opencv/draw.hpp"
#include "av1/corner_detect.h"
#include <opencv2/imgproc.hpp>
#include <chrono>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace std::chrono;

const char *DetectName[] = {
  [Detect::FAST_BEBLID] = "FAST",
  [Detect::FAST_SIFT] = "FAST",
  [Detect::FAST_SURF] = "FAST",
  [Detect::FAST_BRISK] = "FAST",
  [Detect::FAST_BRIEF] = "FAST",
  [Detect::FAST_FREAK] = "FAST",
  [Detect::FAST_DAISY] = "FAST",
  [Detect::AGAST_BEBLID] = "AGAST",
  [Detect::AGAST_SIFT] = "AGAST",
  [Detect::AGAST_SURF] = "AGAST",
  [Detect::AGAST_BRISK] = "AGAST",
  [Detect::AGAST_BRIEF] = "AGAST",
  [Detect::AGAST_FREAK] = "AGAST",
  [Detect::AGAST_DAISY] = "AGAST",
  [Detect::SIFT] = "SIFT",
  [Detect::SURF] = "SURF",
  [Detect::ORB] = "ORB",
  [Detect::BRISK] = "BRISK",
  //                              [Detect::BRIEF] = "BRIEF",
  //                              [Detect::BLOB] = "BLOB",
  //                              [Detect::KAZE] = "KAZE",
  //                              [Detect::AKAZE] = "AKAZE",
  [Detect::FAST_AOM] = "FAST"
};

const char *DescribeName[] = { [Detect::FAST_BEBLID] = "BEBLID",
                               [Detect::FAST_SIFT] = "SIFT",
                               [Detect::FAST_SURF] = "SURF",
                               [Detect::FAST_BRISK] = "BRISK",
                               [Detect::FAST_BRIEF] = "BRIEF",
                               [Detect::FAST_FREAK] = "FREAK",
                               [Detect::FAST_DAISY] = "DAISY",
                               [Detect::AGAST_BEBLID] = "BEBLID",
                               [Detect::AGAST_SIFT] = "SIFT",
                               [Detect::AGAST_SURF] = "SURF",
                               [Detect::AGAST_BRISK] = "BRISK",
                               [Detect::AGAST_BRIEF] = "BRIEF",
                               [Detect::AGAST_FREAK] = "FREAK",
                               [Detect::AGAST_DAISY] = "DAISY",
                               [Detect::SIFT] = "SIFT",
                               [Detect::SURF] = "SURF",
                               [Detect::ORB] = "ORB",
                               [Detect::BRISK] = "BRISK",
                               [Detect::FAST_AOM] = "CROSS" };

const char *MatchName[] = {
  [Match::FLANN_BEST] = "FLANN",
  [Match::FLANN_KNN] = "FLANN",
  //                            [Match::FLANN_RADIUS] = "FLANN",
  [Match::BF_BEST] = "BF",
  [Match::BF_KNN] = "BF",
  //                            [Match::BF_RADIUS] = "BF",
  [Match::BF_AOM] = "BF"
};

const char *ModeName[] = {
  [Match::FLANN_BEST] = "BEST",
  [Match::FLANN_KNN] = "KNN",
  //                           [Match::FLANN_RADIUS] = "RADIUS",
  [Match::BF_BEST] = "BEST",
  [Match::BF_KNN] = "KNN",
  //                           [Match::BF_RADIUS] = "RADIUS",
  [Match::BF_AOM] = "BEST"
};

const char *EstimateName[] = {
  [Estimate::RANSAC] = "RANSAC",   [Estimate::MLESAC] = "MLESAC",
  [Estimate::MSAC] = "MSAC",       [Estimate::PROSAC] = "PROSAC",
  [Estimate::PROMSAC] = "PROMSAC", [Estimate::PROMLESAC] = "PROMLESAC",
  [Estimate::LMEDS] = "LMEDS"
};

string detectName(int d) { return DetectName[d]; }

string matchName(int m) { return MatchName[m]; }

string describeName(int d) { return DescribeName[d]; }

string modeName(int m) { return ModeName[m]; }

string estimateName(int e) { return EstimateName[e]; }

string formatName(Detect detect_type, Match match_type, Estimate estimate_type,
                  int frame) {
  string name = "output/";
  name.append(DetectName[detect_type]);
  name.append("_");
  name.append(DescribeName[detect_type]);
  name.append("_");
  name.append(MatchName[match_type]);
  name.append("_");
  name.append(EstimateName[estimate_type]);
  name.append("_");
  name.append(std::to_string(frame));
  name.append(".png");
  return name;
}

void compute(const Mat &src_frame, const Mat &ref_frame,
             TransformationType transformation_type, Detect detect_type,
             Match match_type, Estimate estimate_type, Stats &stats,
             int frame) {
  auto start = high_resolution_clock::now();

  Mat src_img;
  Mat ref_img;
  cv::cvtColor(src_frame, src_img, cv::COLOR_BGR2GRAY);
  cv::cvtColor(ref_frame, ref_img, cv::COLOR_BGR2GRAY);

  std::vector<KeyPoint> src_keypoints, ref_keypoints;
  Mat src_descriptors, ref_descriptors;
  std::vector<DMatch> good_matches;

  src_keypoints.reserve(MAX_CORNERS * 2);
  ref_keypoints.reserve(MAX_CORNERS * 2);
  src_descriptors.reserve(MAX_CORNERS * 2);
  ref_descriptors.reserve(MAX_CORNERS * 2);
  good_matches.reserve(MAX_CORNERS * 2);

//    string name = formatName(detect_type, match_type, estimate_type, frame);

  int num_src_corners =
      detect(src_img, src_keypoints, src_descriptors, detect_type);

  int num_ref_corners =
      detect(ref_img, ref_keypoints, ref_descriptors, detect_type);

  int num_correspondences = match(src_descriptors, ref_descriptors,
                                  good_matches, match_type, detect_type);

  src_img.release();
  ref_img.release();
  src_descriptors.release();
  ref_descriptors.release();

  stats.src_points_num = num_src_corners;
  stats.ref_points_num = num_ref_corners;
  stats.matches_num = num_correspondences;

  if (num_correspondences == 0) {
    stats.inliers_num = 0;
    stats.outliers_num = 0;
    stats.matches_num = 0;
    stats.inliers_per = 0;

    auto stop = high_resolution_clock::now();
    stats.time = duration_cast<milliseconds>(stop - start).count();

    return;
  }

  Correspondence *correspondences =
      (Correspondence *)malloc(num_correspondences * sizeof(*correspondences));
  Correspondence *ptr = correspondences;

  for (auto &d : good_matches) {
    ptr->x = static_cast<int>(round(src_keypoints[d.queryIdx].pt.x));
    ptr->y = static_cast<int>(round(src_keypoints[d.queryIdx].pt.y));
    ptr->rx = static_cast<int>(round(ref_keypoints[d.trainIdx].pt.x));
    ptr->ry = static_cast<int>(round(ref_keypoints[d.trainIdx].pt.y));
    ptr++;
  }

//    std::vector<DMatch> empty;
//    draw_matches(src_img, ref_img, src_keypoints, ref_keypoints, good_matches, name);

  //  estimate(correspondences, num_correspondences, transformation_type,
  //           estimate_type, stats);

  estimate_av1((int *)correspondences, num_correspondences, transformation_type,
               stats);

  free(correspondences);

  auto stop = high_resolution_clock::now();
  stats.time = duration_cast<milliseconds>(stop - start).count();
}

void av1(const Mat &src_frame, const Mat &ref_frame,
         TransformationType transformation_type, Stats &stats, int frame) {
  Mat src_img;
  Mat ref_img;
  cv::cvtColor(src_frame, src_img, cv::COLOR_BGR2GRAY);
  cv::cvtColor(ref_frame, ref_img, cv::COLOR_BGR2GRAY);

  unsigned char *src_buffer = src_img.data;
  unsigned char *ref_buffer = ref_img.data;

  auto start_1 = high_resolution_clock::now();

  int src_width = src_frame.cols;
  int src_height = src_frame.rows;
  int src_stride = src_width;
  int src_corners[2 * MAX_CORNERS];

  int ref_width = ref_frame.cols;
  int ref_height = ref_frame.rows;
  int ref_stride = ref_width;
  int ref_corners[2 * MAX_CORNERS];

  int num_src_corners = av1_fast_corner_detect(
      src_buffer, src_width, src_height, src_stride, src_corners, MAX_CORNERS);

  int num_ref_corners = av1_fast_corner_detect(
      ref_buffer, ref_width, ref_height, ref_stride, ref_corners, MAX_CORNERS);

  int *correspondences =
      (int *)malloc(num_src_corners * 4 * sizeof(*correspondences));

  int num_correspondences = av1_determine_correspondence(
      src_buffer, (int *)src_corners, num_src_corners, ref_buffer,
      (int *)ref_corners, num_ref_corners, src_width, src_height, src_stride,
      ref_stride, correspondences);

  auto stop_1 = high_resolution_clock::now();

  src_img.release();
  ref_img.release();

  std::vector<DMatch> good_matches;
  std::vector<KeyPoint> src_keypoints, ref_keypoints;

  for (int j = 0; j < num_correspondences; j++) {
    int sx = correspondences[j * 4];
    int sy = correspondences[j * 4 + 1];
    int rx = correspondences[j * 4 + 2];
    int ry = correspondences[j * 4 + 3];

    src_keypoints.emplace_back(sx, sy, 1);
    ref_keypoints.emplace_back(rx, ry, 1);
    good_matches.emplace_back(j, j, 0);
  }

  stats.src_points_num = num_src_corners;
  stats.ref_points_num = num_ref_corners;
  stats.matches_num = num_correspondences;

  if (num_correspondences == 0) {
    free(correspondences);
    stats.inliers_num = 0;
    stats.outliers_num = 0;
    stats.matches_num = 0;
    stats.inliers_per = 0;

    stats.time = duration_cast<milliseconds>(stop_1 - start_1).count();

    return;
  }

  for (int j = 0; j < num_src_corners; j++) {
    src_keypoints.emplace_back(src_corners[2 * j], src_corners[2 * j + 1], 1);
  }

  for (int j = 0; j < num_ref_corners; j++) {
    ref_keypoints.emplace_back(ref_corners[2 * j], ref_corners[2 * j + 1], 1);
  }

  //  std::vector<DMatch> empty;
  //  draw_matches(src_img, ref_img, src_keypoints, ref_keypoints, empty,
  //               "output/aom" + std::to_string(frame) + ".png");

  auto start_2 = high_resolution_clock::now();

  estimate_av1(correspondences, num_correspondences, transformation_type,
               stats);

  free(correspondences);

  auto stop_2 = high_resolution_clock::now();
  stats.time =
      duration_cast<milliseconds>((stop_1 - start_1) + (stop_2 - start_2))
          .count();
}