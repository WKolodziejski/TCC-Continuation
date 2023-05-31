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

void print_latex(Stats &results, int d, int m, ofstream &latex) {
//  float inliers_per =
//      results.matches_num == 0
//          ? 0.0
//          : (float)results.inliers_num / (float)results.matches_num * 100;

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
//  latex << results.outliers_num << " & ";
  latex << std::fixed << setprecision(2) << results.inliers_per << " & ";
  latex << results.time << " \\\\ ";

  if (!(m % 2)) {
    latex << "\\cline{4-11} \n";
  } else if (!(m % 3)) {
    latex << "\\cline{1-11} \n";
  } else {
    latex << "\\cline{3-11} \n";
  }
}

void print_csv(Stats &results, int d, int m, ofstream &csv) {
//  float inliers_per =
//      results.matches_num == 0
//          ? 0.0
//          : (float)results.inliers_num / (float)results.matches_num * 100;

  string inliers = std::to_string(results.inliers_per);
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

void print_cmd(Stats &results, const string &name, int d, int m, int i,
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

void stats_accumulate(Stats &stats_this, Stats &stats) {
  stats_this.time += stats.time;
  stats_this.src_points_num += stats.src_points_num;
  stats_this.ref_points_num += stats.ref_points_num;
  stats_this.matches_num += stats.matches_num;
  stats_this.inliers_num += stats.inliers_num;
  stats_this.outliers_num += stats.outliers_num;
  stats_this.inliers_per += stats.inliers_per;
}

void stats_normalize(Stats &stats, int d) {
  stats.time /= d;
  stats.src_points_num /= d;
  stats.ref_points_num /= d;
  stats.matches_num /= d;
  stats.inliers_num /= d;
  stats.outliers_num /= d;
  stats.inliers_per /= (float) d;
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

string formatNameSolution(Detect detect_type, Match match_type, Estimate estimate_type,
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

string formatName(const string& prefix, int frame) {
  string name = "output/";
  name.append(prefix);
  name.append("_");
  name.append(std::to_string(frame));
  name.append(".png");
  return name;
}

string formatNameCluster(const string& prefix, const string& ver, int frame, int k) {
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