#include "draw.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include "../utils.hpp"
#include "../av1/warp_affine.c"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void draw_matches(Mat &src_img, Mat &ref_img,
                  std::vector<KeyPoint> &src_keypoints,
                  std::vector<KeyPoint> &ref_keypoints,
                  std::vector<DMatch> &good_matches, const string &name) {
  // R = Scalar(0, 0, 255)
  // B = Scalar(255, 0, 0)
  // G = Scalar(0, 255, 0)

  Scalar color_lines = Scalar(0, 0, 255);
  Scalar color_keypoints = Scalar(255, 0, 0);

  Mat img;
  drawMatches(src_img, src_keypoints, ref_img, ref_keypoints, good_matches, img,
              color_lines, color_keypoints, std::vector<char>(),
              DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);
}

void draw_motion_field(Mat &src_img, Mat &ref_img,
                       Correspondence *correspondences, int num_correspondences,
                       const string &name) {
  Mat src;
  Mat ref;
  Mat mask = Mat::zeros(src_img.size(), CV_8UC3);
  cv::cvtColor(src_img, src, cv::COLOR_GRAY2RGB);
  cv::cvtColor(ref_img, ref, cv::COLOR_GRAY2RGB);

  for (int i = 0; i < num_correspondences; i++) {
    Correspondence c = correspondences[i];
    double d = distance(c);

    if (d < 1) {
      circle(mask, Point2f(c.x, c.y), 5, Scalar(0, 0, 255), 1, LINE_AA);
    } else {
      arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                  Scalar(255 - d * 5, d * 5, angle(c)), 1, LINE_AA);
    }
  }

  Mat img;
  addWeighted(src, 0.5, ref, 0.5, 0, img);
  add(img, mask, img);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);
}

void draw_clustered_motion_field(Mat &src_img, Mat &ref_img,
                                 Correspondence *correspondences,
                                 int num_correspondences, const string &name) {
  Mat src;
  Mat ref;
  Mat mask = Mat::zeros(src_img.size(), CV_8UC3);
  cv::cvtColor(src_img, src, cv::COLOR_GRAY2RGB);
  cv::cvtColor(ref_img, ref, cv::COLOR_GRAY2RGB);

  Mat labels;
  Mat centers;
  Mat samples(num_correspondences, 1, CV_32F);

  // TODO: considerar angulo também ?
  for (int i = 0; i < num_correspondences; i++) {
    samples.at<float>(i) = distance(correspondences[i]);
//    samples.at<float>(i, 1) = angle(correspondences[i]);
//    PARECE QUE UTILIZAR O ANGULO NÃO É VANTAJOSO, DEVIDO AO MOVIMENTO DE ZOOM
  }

  double last_error = 0;
  int k = 1;
  int a = 0;
  for (int i = k; i < 10; i++) {
    double error = kmeans(samples, i, labels, TermCriteria(3, 10000, 0.0001), 5,
                          KMEANS_PP_CENTERS, centers);

    if (error < last_error / 2) {
      k = i;
      a++;
    }

    // if attempts > 2 stop increasing k
    if (a > 1) {
      break;
    }

    last_error = error;
  }

  labels.release();
  centers.release();

  fprintf(stderr, "k = %d\n", k);

  kmeans(samples, k, labels, TermCriteria(3, 10000, 0.0001), 5,
         KMEANS_PP_CENTERS, centers);

  vector<Correspondence> clusters[k];

  for (int i = 0; i < num_correspondences; i++) {
    Correspondence c = correspondences[i];
    clusters[labels.at<int>(i)].push_back(c);
  }

  for (int i = 0; i < num_correspondences; i++) {
    Correspondence c = correspondences[i];

    if (distance(c) < 1) {
      circle(mask, Point2f(c.x, c.y), 5, Scalar(0, 0, 255), 1, LINE_AA);
      continue;
    }

    if (clusters[labels.at<int>(i)].size() < 3) {
      arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                  Scalar(0, 0, 255), 1, LINE_AA);
      continue;
    }

    switch (labels.at<int>(i)) {
      case 0:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 0, 0), 1, LINE_AA);
        break;

      case 1:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(0, 255, 0), 1, LINE_AA);
        break;

      case 2:
        //        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
        //                    Scalar(0, 0, 255), 1, LINE_AA);
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(127, 255, 127), 1, LINE_AA);
        break;

      case 3:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(0, 255, 255), 1, LINE_AA);
        break;

      case 4:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 0, 255), 1, LINE_AA);
        break;

      case 5:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 255, 0), 1, LINE_AA);
        break;

      case 6:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 127, 0), 1, LINE_AA);
        break;

      case 7:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(0, 255, 127), 1, LINE_AA);
        break;

      case 8:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(127, 0, 255), 1, LINE_AA);
        break;

      case 9:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(127, 127, 255), 1, LINE_AA);
        break;

      case 10:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 127, 127), 1, LINE_AA);
        break;

        //      case 11:
        //        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
        //                    Scalar(127, 255, 127), 1, LINE_AA);
        //        break;

      default:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 255, 255), 1, LINE_AA);
    }
  }

  Mat img;
  addWeighted(src, 0.5, ref, 0.5, 0, img);
  add(img, mask, img);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);
}

/*
  / a b \  = /   1       0    \ * / 1+alpha  beta \
  \ c d /    \ gamma  1+delta /   \    0      1   /
  where a, b, c, d are wmmat[2], wmmat[3], wmmat[4], wmmat[5] respectively.
 */
double draw_warped(const Mat &src_img, const Mat &ref_img, const double mat[8], const string &name) {
  Mat warp_mat = Mat::zeros(2, 3, CV_64FC1);
  warp_mat.at<double>(0) = mat[2];  // scale x
  warp_mat.at<double>(1) = mat[3];  // rot +
  warp_mat.at<double>(2) = mat[0];  // trans x
  warp_mat.at<double>(3) = mat[4];  // rot -
  warp_mat.at<double>(4) = mat[5];  // scale y
  warp_mat.at<double>(5) = mat[1];  // trans y

  Mat img;
  warpAffine(src_img, img, warp_mat, src_img.size(), INTER_AREA);

  addWeighted(img, 0.5, ref_img, 0.5, 0, img);

  Mat error_img;
  subtract(img, ref_img, error_img);

  multiply(error_img, error_img, error_img);

  double error = sum(error_img)[0];

  fprintf(stderr, "e = %f\n", error);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);

  return error;
}