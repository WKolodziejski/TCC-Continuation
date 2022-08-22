#include <iostream>
#include <fstream>
#include <iomanip>
#include "william.hpp"

inline void print_latex(Stats &results, int d, int m, ofstream &latex) {
  float inliers_per =
      results.matches_num == 0
          ? 0.0
          : (float)results.inliers_num / (float)results.matches_num * 100;

  if (m == BF_AOM) {
    latex << detectName(d) << " & ";
    latex << describeName(d) << " & ";
  } else if (!(m % 4)) {
    latex << "\\multirow{4}{*}{" << detectName(d) << "} & ";
    latex << "\\multirow{4}{*}{" << describeName(d) << "} & ";
  } else {
    latex << " & & ";
  }

  if (m == BF_AOM) {
    latex << matchName(m) << " & ";
  } else if (!(m % 2)) {
    latex << "\\multirow{2}{*}{" << matchName(m) << "} & ";
  } else {
    latex << " & ";
  }

  latex << modeName(m) << " & ";
  latex << results.src_points_num << " & ";
  latex << results.ref_points_num << " & ";
  latex << results.matches_num << " & ";
  latex << results.inliers_num << " & ";
  latex << results.outliers_num << " & ";
  latex << std::fixed << setprecision(2) << inliers_per << " & ";
  latex << results.time << " \\\\ ";

  if (!(m % 2)) {
    latex << "\\cline{4-11} \n";
  } else if (!(m % 3)) {
    latex << "\\cline{1-11} \n";
  } else {
    latex << "\\cline{3-11} \n";
  }
}

inline void print_csv(Stats &results, int d, int m, ofstream &csv) {
  float inliers_per =
      results.matches_num == 0
          ? 0.0
          : (float)results.inliers_num / (float)results.matches_num * 100;

  string inliers = std::to_string(inliers_per);
  std::replace(inliers.begin(), inliers.end(), '.', ',');

  csv << detectName(d) << ";";
  csv << describeName(d) << ";";
  csv << matchName(m) << ";";
  csv << modeName(m) << ";";
  csv << results.src_points_num << ";";
  csv << results.ref_points_num << ";";
  csv << results.matches_num << ";";
  csv << results.inliers_num << ";";
  csv << results.outliers_num << ";";
  csv << inliers << ";";
  csv << results.time << "\n";
}

inline void print_cmd(Stats &results, const string &name, int d, int m, int i,
                      int f, int t) {
  fprintf(stderr,
          "%d\t|"
          "\t%s\t|"
          "\t%d\t|"
          "\t%d\t|"
          "\t%s\t|"
          "\t%s\t|"
          "\t%s\t|"
          "\t%s\t|"
          "\t%.2f\t|"
          "\t%lu ms\n",
          t, name.substr(0, name.length() < 8 ? name.length() : 8).c_str(), i,
          f, detectName(d).c_str(), describeName(d).c_str(),
          matchName(m).c_str(), modeName(m).c_str(), results.inliers_per,
          results.time);
}

inline void stats_accumulate(Stats &stats_this, Stats &stats) {
  stats_this.time += stats.time;
  stats_this.src_points_num += stats.src_points_num;
  stats_this.ref_points_num += stats.ref_points_num;
  stats_this.matches_num += stats.matches_num;
  stats_this.inliers_num += stats.inliers_num;
  stats_this.outliers_num += stats.outliers_num;
}

inline void stats_normalize(Stats &stats, int d) {
  stats.time /= d;
  stats.src_points_num /= d;
  stats.ref_points_num /= d;
  stats.matches_num /= d;
  stats.inliers_num /= d;
  stats.outliers_num /= d;
}
