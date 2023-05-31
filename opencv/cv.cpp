#include "../william.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

#define RATIO_THRESH 0.75
#define MAX_CORNERS 4096

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace std::chrono;

// default
inline int detect_default(Mat &img, vector<KeyPoint> &kpts, Mat &desc,
                          Detect type) {
  if (type == Detect::FAST_SIFT) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(18);
    detector->detect(img, kpts);

    Ptr<Feature2D> sift = SIFT::create();
    sift->compute(img, kpts, desc);

  } else if (type == Detect::FAST_SURF) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(18);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = SURF::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_BEBLID) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(18);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = BEBLID::create(5);
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_BRISK) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(18);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = BRISK::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_BRIEF) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(18);
    detector->detect(img, kpts);

    Ptr<BriefDescriptorExtractor> descriptor =
        BriefDescriptorExtractor::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_FREAK) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(18);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = FREAK::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_DAISY) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(18);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = DAISY::create();
    descriptor->compute(img, kpts, desc);

    //  } else if (type == Detect::AGAST_SIFT) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(18);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> sift = SIFT::create();
    //    sift->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_SURF) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(18);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> sift = SURF::create();
    //    sift->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_BEBLID) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(18);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> beblid = BEBLID::create(5);
    //    beblid->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_BRISK) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(18);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> brisk = BRISK::create();
    //    brisk->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_BRIEF) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(18);
    //    detector->detect(img, kpts);
    //
    //    Ptr<BriefDescriptorExtractor> brief =
    //    BriefDescriptorExtractor::create(); brief->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_FREAK) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(18);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> brisk = FREAK::create();
    //    brisk->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_DAISY) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(18);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> brisk = DAISY::create();
    //    brisk->compute(img, kpts, desc);

  } else if (type == Detect::SURF) {
    Ptr<Feature2D> surf = SURF::create();
    surf->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::SIFT) {
    Ptr<Feature2D> sift = SIFT::create();
    sift->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::ORB) {
    Ptr<Feature2D> orb = ORB::create(MAX_CORNERS * 2);
    orb->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::BRISK) {
    Ptr<Feature2D> brisk = BRISK::create();
    brisk->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::KAZE) {
    Ptr<Feature2D> kaze = KAZE::create();
    kaze->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::AKAZE) {
    Ptr<Feature2D> akaze = AKAZE::create();
    akaze->detectAndCompute(img, noArray(), kpts, desc);

  } else {
    assert(0);
  }

  return static_cast<int>(kpts.size());
}

// custom
inline int detect_custom(Mat &img, vector<KeyPoint> &kpts, Mat &desc,
                         Detect type) {
  if (type == Detect::FAST_SIFT) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
    detector->detect(img, kpts);

    Ptr<Feature2D> sift = SIFT::create();
    sift->compute(img, kpts, desc);

  } else if (type == Detect::FAST_SURF) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = SURF::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_BEBLID) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(60);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = BEBLID::create(5);
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_BRISK) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = BRISK::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_BRIEF) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
    detector->detect(img, kpts);

    Ptr<BriefDescriptorExtractor> descriptor =
        BriefDescriptorExtractor::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_FREAK) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = FREAK::create();
    descriptor->compute(img, kpts, desc);

  } else if (type == Detect::FAST_DAISY) {
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
    detector->detect(img, kpts);

    Ptr<Feature2D> descriptor = DAISY::create();
    descriptor->compute(img, kpts, desc);

    //  } else if (type == Detect::AGAST_SIFT) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(40);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> sift = SIFT::create();
    //    sift->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_SURF) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(40);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> sift = SURF::create();
    //    sift->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_BEBLID) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(40);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> beblid = BEBLID::create(5);
    //    beblid->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_BRISK) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(40);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> brisk = BRISK::create();
    //    brisk->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_BRIEF) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(40);
    //    detector->detect(img, kpts);
    //
    //    Ptr<BriefDescriptorExtractor> brief =
    //    BriefDescriptorExtractor::create(); brief->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_FREAK) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(40);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> brisk = FREAK::create();
    //    brisk->compute(img, kpts, desc);
    //
    //  } else if (type == Detect::AGAST_DAISY) {
    //    Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(40);
    //    detector->detect(img, kpts);
    //
    //    Ptr<Feature2D> brisk = DAISY::create();
    //    brisk->compute(img, kpts, desc);

  } else if (type == Detect::SURF) {
    Ptr<Feature2D> surf = SURF::create(200);
    surf->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::SIFT) {
    Ptr<Feature2D> sift = SIFT::create(MAX_CORNERS);
    sift->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::ORB) {
    Ptr<Feature2D> orb = ORB::create(MAX_CORNERS);
    orb->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::BRISK) {
    Ptr<Feature2D> brisk = BRISK::create(60);
    brisk->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::KAZE) {
    Ptr<Feature2D> kaze = KAZE::create(false, false, 0.002f);
    kaze->detectAndCompute(img, noArray(), kpts, desc);

  } else if (type == Detect::AKAZE) {
    Ptr<Feature2D> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.002f);
    akaze->detectAndCompute(img, noArray(), kpts, desc);

  } else {
    assert(0);
  }

  return static_cast<int>(kpts.size());
}

inline void knn(Mat &desc1, Mat &desc2, vector<DMatch> &matches,
                Match match_type, bool flann_compat) {
  if (desc1.total() < 2 || desc2.total() < 2) return;

  vector<vector<DMatch>> vmatches;

  if (match_type == Match::BF_KNN) {
    Ptr<DescriptorMatcher> matcher =
        DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    matcher->knnMatch(desc1, desc2, vmatches, 2);

  } else if (match_type == Match::FLANN_KNN) {
    if (flann_compat) {
      FlannBasedMatcher matcher(new flann::LshIndexParams(12, 20, 2));
      matcher.knnMatch(desc1, desc2, vmatches, 2);
    } else {
      Ptr<DescriptorMatcher> matcher =
          DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
      matcher->knnMatch(desc1, desc2, vmatches, 2);
    }
  } else {
    assert(0);
  }

  for (auto &knn_match : vmatches) {
    if (knn_match.empty()) {
      continue;
    }

    if (knn_match[0].distance < RATIO_THRESH * knn_match[1].distance) {
      matches.push_back(knn_match[0]);
    }
  }
}

inline void best(Mat &desc1, Mat &desc2, vector<DMatch> &matches,
                 Match match_type, bool flann_compat) {
  if (match_type == Match::BF_BEST) {
    Ptr<DescriptorMatcher> matcher =
        DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    matcher->match(desc1, desc2, matches);

  } else if (match_type == Match::FLANN_BEST) {
    if (flann_compat) {
      FlannBasedMatcher matcher(new flann::LshIndexParams(12, 20, 2));
      matcher.match(desc1, desc2, matches);
    } else {
      Ptr<DescriptorMatcher> matcher =
          DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
      matcher->match(desc1, desc2, matches);
    }
  } else {
    assert(0);
  }
}

inline int detect(Mat &img, vector<KeyPoint> &kpts, Mat &desc, Detect type,
                  bool custom) {
  if (custom) {
    detect_custom(img, kpts, desc, type);
  } else {
    detect_default(img, kpts, desc, type);
  }
}

inline int match(Mat &desc1, Mat &desc2, vector<DMatch> &matches,
                 Match match_type, Detect detect_type) {
  bool flann_compat =
      detect_type == Detect::FAST_BEBLID || detect_type == Detect::FAST_BRISK ||
      detect_type == Detect::FAST_BRIEF || detect_type == Detect::FAST_FREAK ||
      //      detect_type == Detect::AGAST_BEBLID ||
      //      detect_type == Detect::AGAST_BRISK ||
      //      detect_type == Detect::AGAST_BRIEF ||
      //      detect_type == Detect::AGAST_FREAK ||
      detect_type == Detect::ORB || detect_type == Detect::BRISK ||
      detect_type == Detect::AKAZE;

  if (match_type == FLANN_KNN || match_type == BF_KNN) {
    knn(desc1, desc2, matches, match_type, flann_compat);
  } else if (match_type == FLANN_BEST || match_type == BF_BEST) {
    best(desc1, desc2, matches, match_type, flann_compat);
  } else {
    assert(0);
  }

  std::sort(matches.begin(), matches.end());

  // Need since MotionParams->inliers has maximum size of MAX_CORNERS
  while (matches.size() > MAX_CORNERS) {
    matches.pop_back();
  }

  return static_cast<int>(matches.size());
}