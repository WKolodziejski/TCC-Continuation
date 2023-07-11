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
  csv << "video;frame;error;seg_error;k_error;seg_gain;k_gain;seg_percent;k_percent\n";

  for (int v = 2; v < argc; v++) {
    string name = argv[v];
    VideoCapture video(name);
    cv::Mat src_frame;
    cv::Mat ref_frame;
    name = name.substr(name.find_last_of('/') + 1);
    name = name.substr(0, name.find_last_of('.'));

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
      stats[f].seg_gain = stats[f].error - stats[f].seg_error;
      stats[f].seg_percent = (double) stats[f].seg_gain / (double) stats[f].error;

      stats[f].k_gain = stats[f].error - stats[f].k_error;
      stats[f].k_percent = (double) stats[f].k_gain / (double) stats[f].error;

      fprintf(stderr, "frame %d\n", f);
      fprintf(stderr, "seg_gain: %d\n", stats[f].seg_gain);
      fprintf(stderr, "seg_percent: %f\n", stats[f].seg_percent);
      fprintf(stderr, "k_gain: %d\n", stats[f].k_gain);
      fprintf(stderr, "k_percent: %f\n", stats[f].k_percent);

      string seg_percent = std::to_string(stats[f].seg_percent);
      string k_percent = std::to_string(stats[f].k_percent);
      std::replace(seg_percent.begin(), seg_percent.end(), '.', ',');
      std::replace(k_percent.begin(), k_percent.end(), '.', ',');

      csv << name << ";";
      csv << f << ";";
      csv << stats[f].error << ";";
      csv << stats[f].seg_error << ";";
      csv << stats[f].k_error << ";";
      csv << stats[f].seg_gain << ";";
      csv << stats[f].k_gain << ";";
      csv << seg_percent << ";";
      csv << k_percent << "\n";
    }
  }

  csv.close();
}