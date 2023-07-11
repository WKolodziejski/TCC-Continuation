#include <opencv2/videoio.hpp>
#include "utils.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

#define FRAMES 7

// args: <output_folder> <input_videos...>
int main(int argc, char *argv[]) {
  string folder = argv[1];
  ofstream csv;
  csv.open(folder.append("results.csv"));
  csv << "video;frame;error;seg_error;gain;percent\n";

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

    fprintf(stderr, "%s\n", name.c_str());

    for (int f = 0; f < FRAMES; f++) {
      stats[f].gain = stats[f].non_segmented_error - stats[f].segmented_error;
      stats[f].percent = (double) stats[f].gain / (double) stats[f].non_segmented_error;

      fprintf(stderr, "frame %d\n", f);
      fprintf(stderr, "gain: %d\n", stats[f].gain);
      fprintf(stderr, "percent: %f\n", stats[f].percent);

      string percent = std::to_string(stats[f].percent);
      std::replace(percent.begin(), percent.end(), '.', ',');

      csv << name << ";";
      csv << f << ";";
      csv << stats[f].non_segmented_error << ";";
      csv << stats[f].segmented_error << ";";
      csv << stats[f].gain << ";";
      csv << percent << "\n";
    }
  }

  csv.close();
}