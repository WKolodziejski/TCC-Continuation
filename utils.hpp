#ifndef UTILS_H
#define UTILS_H

#include "solution.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

#ifdef __cplusplus
extern "C" {
#endif

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

double distance(Correspondence &c);

double angle(Correspondence &c);

Mat parse_affine_mat(const double mat[8]);

string detectName(int d);

string matchName(int m);

string describeName(int d);

string modeName(int m);

string estimateName(int e);

string formatNameSolution(Detect detect_type, Match match_type,
                          Estimate estimate_type, int frame);

string formatName(const string &prefix, int frame);

string formatNameCluster(const string &prefix, const string &ver, int frame,
                         int k);

void calc_seg_error(const Mat &src_img, const Mat &ref_img,
                     const double mat[8], int x, int y, MatrixMap **map, int k,
                     bool zero_motion);

double calc_error(const Mat &src_img, const Mat &ref_img,
                           const double mat[8]);

#ifdef __cplusplus
}
#endif

#endif  // UTILS_H
