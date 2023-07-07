#include <opencv2/videoio.hpp>
#include "utils.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std::chrono;

#define ITERATIONS 1
#define FRAMES 7

// args: <output_folder> <input_videos...>
int main(int argc, char *argv[]) {
  fprintf(stderr, "%s", cv::getBuildInformation().c_str());

  string folder = argv[1];

  for (int v = 2; v < argc; v++) {
    string name = argv[v];
    VideoCapture video(name);
    cv::Mat src_frame;
    cv::Mat ref_frame;
    name = name.substr(name.find_last_of('/') + 1);
    name = name.substr(0, name.find_last_of('.'));

    for (int s = name.length(); s < 8; s++) name.append(" ");

    auto start_this = high_resolution_clock::now();

    if (!video.isOpened()) exit(-1);
    if (!video.read(src_frame)) exit(-1);

    Stats stats_all[FRAMES][ITERATIONS][FAST_AOM + 1][BF_AOM + 1];

    for (int f = 0; f < FRAMES; f++) {
      if (!video.read(ref_frame)) break;

      compute(src_frame, ref_frame, ROTZOOM, Detect::FAST_BEBLID,
              Match::FLANN_KNN, Estimate::RANSAC,
              stats_all[f][0][Detect::FAST_BEBLID][Estimate::RANSAC], f);
    }

    video.release();
  }
}
