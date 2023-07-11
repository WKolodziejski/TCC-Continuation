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