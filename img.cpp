#include <opencv2/videoio.hpp>
#include "opencv/cv.cpp"
#include "opencv/draw.hpp"
#include <opencv2/imgproc.hpp>
#include "av1/ransac.h"
#include "av1/corner_detect.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std::chrono;

// Usado para gerar imagens mais facilmente
int main(int argc, char *argv[]) {
  string folder = argv[1];

  VideoCapture video(argv[2]);
  cv::Mat src_frame;
  cv::Mat ref_frame;

  Detect detect_type = Detect::FAST_DAISY;
  Match match_type = Match::FLANN_KNN;
  Estimate estimate_type = Estimate::RANSAC_AOM;

  if (!video.isOpened()) return -1;
  if (!video.read(src_frame)) return -1;

  int f = 0;

  while (f++ < 7)
    if (!video.read(ref_frame)) return -1;

  string name = formatNameSolution(detect_type, match_type, estimate_type, 0);

    Mat src_img;
    Mat ref_img;
    cv::cvtColor(src_frame, src_img, cv::COLOR_BGR2GRAY);
    cv::cvtColor(ref_frame, ref_img, cv::COLOR_BGR2GRAY);

    std::vector<KeyPoint> src_keypoints, ref_keypoints;
    Mat src_descriptors, ref_descriptors;
    std::vector<DMatch> good_matches;

    int num_src_corners =
        detect(src_img, src_keypoints, src_descriptors, detect_type, false);

    int num_ref_corners =
        detect(ref_img, ref_keypoints, ref_descriptors, detect_type, false);

    int num_correspondences = match(src_descriptors, ref_descriptors,
                                    good_matches, match_type, detect_type);

    Correspondence *correspondences =
        (Correspondence *)malloc(num_correspondences *
        sizeof(*correspondences));
    Correspondence *ptr = correspondences;

    for (auto &d : good_matches) {
      ptr->x = static_cast<int>(round(src_keypoints[d.queryIdx].pt.x));
      ptr->y = static_cast<int>(round(src_keypoints[d.queryIdx].pt.y));
      ptr->rx = static_cast<int>(round(ref_keypoints[d.trainIdx].pt.x));
      ptr->ry = static_cast<int>(round(ref_keypoints[d.trainIdx].pt.y));
      ptr++;
    }

//  Mat src_img;
//  Mat ref_img;
//  cv::cvtColor(src_frame, src_img, cv::COLOR_BGR2GRAY);
//  cv::cvtColor(ref_frame, ref_img, cv::COLOR_BGR2GRAY);
//
//  unsigned char *src_buffer = src_img.data;
//  unsigned char *ref_buffer = ref_img.data;
//
//  int src_width = src_frame.cols;
//  int src_height = src_frame.rows;
//  int src_stride = src_width;
//  int src_corners[2 * MAX_CORNERS];
//
//  int ref_width = ref_frame.cols;
//  int ref_height = ref_frame.rows;
//  int ref_stride = ref_width;
//  int ref_corners[2 * MAX_CORNERS];
//
//  int num_src_corners = av1_fast_corner_detect(
//      src_buffer, src_width, src_height, src_stride, src_corners, MAX_CORNERS);
//
//  int num_ref_corners = av1_fast_corner_detect(
//      ref_buffer, ref_width, ref_height, ref_stride, ref_corners, MAX_CORNERS);
//
//  int *correspondences =
//      (int *)malloc(num_src_corners * 4 * sizeof(*correspondences));
//
//  int num_correspondences = av1_determine_correspondence(
//      src_buffer, (int *)src_corners, num_src_corners, ref_buffer,
//      (int *)ref_corners, num_ref_corners, src_width, src_height, src_stride,
//      ref_stride, correspondences);
//
//  std::vector<DMatch> good_matches;
//  std::vector<KeyPoint> src_keypoints, ref_keypoints;
//
//  for (int j = 0; j < num_correspondences; j++) {
//    int sx = correspondences[j * 4];
//    int sy = correspondences[j * 4 + 1];
//    int rx = correspondences[j * 4 + 2];
//    int ry = correspondences[j * 4 + 3];
//
//    src_keypoints.emplace_back(sx, sy, 7);
//    ref_keypoints.emplace_back(rx, ry, 7);
//    good_matches.emplace_back(j, j, 0);
//  }
//
//  for (int j = 0; j < num_src_corners; j++) {
//    src_keypoints.emplace_back(src_corners[2 * j], src_corners[2 * j + 1], 1);
//  }
//
//  for (int j = 0; j < num_ref_corners; j++) {
//    ref_keypoints.emplace_back(ref_corners[2 * j], ref_corners[2 * j + 1], 1);
//  }

  //----------------------------------------

  int num_inliers_by_motion[1];
  MotionModel params_by_motion[1];
  params_by_motion[0].num_inliers = 0;
  params_by_motion[0].inliers = static_cast<int *>(
      malloc(sizeof(*(params_by_motion[0].inliers)) * 2 * num_correspondences));

  RansacFunc ransac = av1_get_ransac_type(ROTZOOM);
  ransac((int *)correspondences, num_correspondences, num_inliers_by_motion,
         params_by_motion, 1);

  std::vector<DMatch> inliers;

//  for (int i = 0; i < params_by_motion[0].num_inliers; i++) {
//    int idx = params_by_motion[0].inliers[i];
//
//    inliers.push_back(good_matches[idx]);
//  }

  Mat src_input = Mat::zeros(src_img.rows, src_img.cols, src_img.type());
  Mat ref_input = Mat::zeros(ref_img.rows, ref_img.cols, ref_img.type());
//  std::vector<KeyPoint> keypoints;

  draw_matches(src_img, ref_img, src_keypoints, ref_keypoints, inliers,
               name);

  free(params_by_motion[0].inliers);
}