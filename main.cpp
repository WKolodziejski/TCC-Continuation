#include <opencv2/videoio.hpp>
#include "utils.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

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

    if (!video.isOpened()) exit(-1);
    if (!video.read(src_frame)) exit(-1);

    Stats stats[FRAMES];

    for (int f = 0; f < FRAMES; f++) {
      if (!video.read(ref_frame)) break;

      compute(src_frame, ref_frame, ROTZOOM, Detect::FAST_BEBLID,
              Match::FLANN_KNN, Estimate::RANSAC, stats[f], f);
    }

    video.release();

    stats_percent(std::vector<Stats>(stats, stats + FRAMES));
  }
}