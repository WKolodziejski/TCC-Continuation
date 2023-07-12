#include <opencv2/imgproc.hpp>
#include "utils.hpp"

double distance(Correspondence &c) {
  double x = pow(c.x - c.rx, 2);
  double y = pow(c.y - c.ry, 2);

  return sqrt(x + y);
}

double angle(Correspondence &c) {
  double x = c.x - c.rx;
  double y = c.y - c.ry;

  return atan2(y, x) * 180 / 3.14159265;
}

/*
  / a b \  = /   1       0    \ * / 1+alpha  beta \
  \ c d /    \ gamma  1+delta /   \    0      1   /
  where a, b, c, d are wmmat[2], wmmat[3], wmmat[4], wmmat[5] respectively.
 */
Mat parse_affine_mat(const double mat[8]) {
  Mat warp_mat = Mat::zeros(2, 3, CV_64FC1);

  warp_mat.at<double>(0) = mat[2];  // scale x
  warp_mat.at<double>(1) = mat[3];  // rot +
  warp_mat.at<double>(2) = mat[0];  // trans x
  warp_mat.at<double>(3) = mat[4];  // rot -
  warp_mat.at<double>(4) = mat[5];  // scale y
  warp_mat.at<double>(5) = mat[1];  // trans y

  return warp_mat;
}

const char *DetectName[] = {
  [Detect::FAST_BEBLID] = "FAST", [Detect::FAST_SIFT] = "FAST",
  [Detect::FAST_SURF] = "FAST",   [Detect::FAST_BRISK] = "FAST",
  [Detect::FAST_BRIEF] = "FAST",  [Detect::FAST_FREAK] = "FAST",
  [Detect::FAST_DAISY] = "FAST",  [Detect::SIFT] = "SIFT",
  [Detect::SURF] = "SURF",        [Detect::ORB] = "ORB",
  [Detect::BRISK] = "BRISK",      [Detect::KAZE] = "KAZE",
  [Detect::AKAZE] = "AKAZE",      [Detect::FAST_AOM] = "FAST"
};

const char *DescribeName[] = {
  [Detect::FAST_BEBLID] = "BEBLID", [Detect::FAST_SIFT] = "SIFT",
  [Detect::FAST_SURF] = "SURF",     [Detect::FAST_BRISK] = "BRISK",
  [Detect::FAST_BRIEF] = "BRIEF",   [Detect::FAST_FREAK] = "FREAK",
  [Detect::FAST_DAISY] = "DAISY",   [Detect::SIFT] = "SIFT",
  [Detect::SURF] = "SURF",          [Detect::ORB] = "ORB",
  [Detect::BRISK] = "BRISK",        [Detect::KAZE] = "KAZE",
  [Detect::AKAZE] = "AKAZE",        [Detect::FAST_AOM] = "CROSS"
};

const char *MatchName[] = { [Match::FLANN_BEST] = "FLANN",
                            [Match::FLANN_KNN] = "FLANN",
                            [Match::BF_BEST] = "BF",
                            [Match::BF_KNN] = "BF",
                            [Match::BF_AOM] = "BF" };

const char *ModeName[] = { [Match::FLANN_BEST] = "BEST",
                           [Match::FLANN_KNN] = "KNN",
                           [Match::BF_BEST] = "BEST",
                           [Match::BF_KNN] = "KNN",
                           [Match::BF_AOM] = "BEST" };

const char *EstimateName[] = {
  [Estimate::RANSAC] = "RANSAC",   [Estimate::MLESAC] = "MLESAC",
  [Estimate::MSAC] = "MSAC",       [Estimate::PROSAC] = "PROSAC",
  [Estimate::PROMSAC] = "PROMSAC", [Estimate::PROMLESAC] = "PROMLESAC",
  [Estimate::LMEDS] = "LMEDS",     [Estimate::RANSAC_AOM] = "AOM"
};

string detectName(int d) { return DetectName[d]; }

string matchName(int m) { return MatchName[m]; }

string describeName(int d) { return DescribeName[d]; }

string modeName(int m) { return ModeName[m]; }

string estimateName(int e) { return EstimateName[e]; }

string formatNameSolution(Detect detect_type, Match match_type,
                          Estimate estimate_type, int frame) {
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

string formatName(const string &prefix, int frame) {
  string name = "output/";
  name.append(prefix);
  name.append("_");
  name.append(std::to_string(frame));
  name.append(".png");
  return name;
}

string formatNameCluster(const string &prefix, const string &ver, int frame,
                         int k) {
  string name = "output/";
  name.append(ver);
  name.append("_");
  name.append(prefix);
  name.append("_");
  name.append(std::to_string(frame));
  name.append("_");
  name.append(std::to_string(k));
  name.append(".png");
  return name;
}

Mat calc_residual_img(const Mat &src_img, const Mat &ref_img, const double mat[8]) {
  Mat warp_mat = parse_affine_mat(mat);

  Mat src_img_signed;
  Mat ref_img_signed;

  src_img.convertTo(src_img_signed, CV_32F);
  ref_img.convertTo(ref_img_signed, CV_32F);

  Mat warped_img;
  warpAffine(src_img_signed, warped_img, warp_mat, src_img_signed.size(), INTER_AREA);

  Mat error_img_signed;
  subtract(warped_img, ref_img_signed, error_img_signed);
  multiply(error_img_signed, error_img_signed, error_img_signed);

  Mat error_img_unsigned;
  error_img_signed.convertTo(error_img_unsigned, CV_8U);

  return error_img_unsigned;
}

void calc_seg_error(const Mat &src_img, const Mat &ref_img, const double mat[8],
                    int x, int y, MatrixMap **map, int k, bool zero_motion) {
  Mat warp_mat = Mat::zeros(2, 3, CV_64FC1);
  warp_mat.at<double>(0) = mat[2];  // scale x
  warp_mat.at<double>(1) = mat[3];  // rot +
  warp_mat.at<double>(2) = mat[0];  // trans x
  warp_mat.at<double>(3) = mat[4];  // rot -
  warp_mat.at<double>(4) = mat[5];  // scale y
  warp_mat.at<double>(5) = mat[1];  // trans y

  Mat error_img = calc_residual_img(src_img, ref_img, mat);

  for (int xi = 0; xi < x; xi++) {
    for (int yj = 0; yj < y; yj++) {
      Rect roi = Rect(xi * WARP_BLOCK_SIZE, yj * WARP_BLOCK_SIZE,
                      WARP_BLOCK_SIZE, WARP_BLOCK_SIZE);
      Mat cropped = error_img(roi);

      double error = sum(cropped)[0];

      if (error < map[xi][yj].error) {
        map[xi][yj].error = error;
        map[xi][yj].k = k;
        map[xi][yj].zero_motion = zero_motion;

        // Copia a matriz
        for (int j = 0; j < 8; j++) {
          map[xi][yj].mat[j] = mat[j];
        }
      }
    }
  }
}

double calc_error(const Mat &src_img, const Mat &ref_img, const double mat[8]) {
  Mat error_img = calc_residual_img(src_img, ref_img, mat);

  double error = sum(error_img)[0];

  return error;
}