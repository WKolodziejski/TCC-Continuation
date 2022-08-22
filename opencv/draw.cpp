#include "draw.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void draw_matches(Mat &src_img, Mat &ref_img,
                  std::vector<KeyPoint> &src_keypoints,
                  std::vector<KeyPoint> &ref_keypoints,
                  std::vector<DMatch> &good_matches, const string &name) {
  Mat img_matches;
  drawMatches(src_img, src_keypoints, ref_img, ref_keypoints, good_matches,
              img_matches, Scalar(0, 255, 0), Scalar(255, 0, 0),
              std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

//  Mat img_outliers;
//  drawMatches(src_img, src_keypoints, ref_img, ref_keypoints, good_matches,
//              img_matches, Scalar(0, 255, 0), Scalar(0, 0, 255),
//              std::vector<char>(), DrawMatchesFlags::DEFAULT);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img_matches, params);
}