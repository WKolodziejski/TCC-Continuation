#ifndef DRAW_H

#include "../av1/corner_match.h"
#include "../solution.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>

#define DRAW_H

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void draw_k_warped_image(const Mat &src_img, const Mat &ref_img, int x, int y,
                         MatrixMap **map, const string &inv_name, const string &fwd_name);

void draw_matches(Mat &src_img, Mat &ref_img,
                  std::vector<KeyPoint> &src_keypoints,
                  std::vector<KeyPoint> &ref_keypoints,
                  std::vector<DMatch> &good_matches, const string &name);

void draw_motion_field(Mat &src_img, Mat &ref_img,
                       Correspondence *correspondences, int num_correspondences,
                       const string &name);

void draw_clustered_motion_field(Mat &src_img, Mat &ref_img,
                                 Correspondence *correspondences,
                                 int num_correspondences, const string &name);

void draw_warped(const Mat &src_img, const Mat &ref_img, const double mat[8],
                   const string &name);

// #ifdef __cplusplus
//  extern "C" {
// #endif
//
//  void draw(unsigned char *src_buffer, int src_width, int src_height,
//           int src_stride, unsigned char *ref_buffer, int ref_width,
//           int ref_height, int ref_stride, int *correspondences,
//           int num_correspondences);
//
// #ifdef __cplusplus
// }
// #endif

#endif  // DRAW_H
