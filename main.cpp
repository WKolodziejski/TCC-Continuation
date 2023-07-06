#include <opencv2/videoio.hpp>
#include "utils.hpp"
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std::chrono;

#define ITERATIONS 1
#define FRAMES 7

// args: <output_folder> <input_videos...>
int main(int argc, char *argv[]) {
  auto start_global = high_resolution_clock::now();
  Stats stats_global[FAST_AOM + 1][BF_AOM + 1];
  string folder = argv[1];

  fprintf(stderr, "\n--STARTING FOR\t%d ITERATIONS\t%d FRAMES\n", ITERATIONS,
          FRAMES);

  fprintf(stderr,
          "\nTHREAD\t|"
          "\tSEQUENCE\t|"
          "\tITER\t|"
          "\tFRAME\t|"
          "\tDETECT\t|"
          "\tDESCRIBE|"
          "\tMATCH\t|"
          "\tMODE\t|"
          "\tINLIERS\t|"
          "\tTIME\n");

//---------------------Launch threads for each video-------------------------

  omp_set_dynamic(0);
  omp_set_num_threads(12);

  #pragma omp parallel default(none) \
    shared(argc, argv, folder, stats_global, start_global, stderr)
  {
    #pragma omp for schedule(static)
    for (int v = 2; v < argc; v++) {
      string name = argv[v];
      VideoCapture video(name);
      cv::Mat src_frame;
      cv::Mat ref_frame;
      name = name.substr(name.find_last_of('/') + 1);
      name = name.substr(0, name.find_last_of('.'));

      for (int s = name.length(); s < 8; s++) name.append(" ");

      auto start_this = high_resolution_clock::now();

      if (!video.isOpened()) continue;
      if (!video.read(src_frame)) continue;

      Stats stats_all[FRAMES][ITERATIONS][FAST_AOM + 1][BF_AOM + 1];

      for (int f = 0; f < FRAMES; f++) {
        if (!video.read(ref_frame)) break;

        for (int i = 0; i < ITERATIONS; i++) {
          for (int d = Detect::FAST_BEBLID; d < Detect::FAST_SIFT; d++) {
            for (int m = Match::FLANN_KNN; m < Match::BF_BEST; m++) {
              compute(src_frame, ref_frame, ROTZOOM, static_cast<Detect>(d),
                      static_cast<Match>(m), Estimate::RANSAC,
                      stats_all[f][i][d][m], f);

//              print_cmd(stats_all[f][i][d][m], name, d, m, i, f,
//                        omp_get_thread_num());
            }
          }

//          av1(src_frame, ref_frame, ROTZOOM, stats_all[f][i][FAST_AOM][BF_AOM],
//              f);
//
//          print_cmd(stats_all[f][i][FAST_AOM][BF_AOM], name, FAST_AOM, BF_AOM,
//                    i, f, omp_get_thread_num());
        }
      }

      video.release();

      //---------------------------Initialize files----------------------------
      ofstream csv_this;
      ofstream latex_this;
      string s(folder + name);
      s.append("_f")
          .append(std::to_string(FRAMES))
          .append("_i")
          .append(std::to_string(ITERATIONS))
          .append("_t")
          .append(std::to_string(omp_get_thread_num()));
      csv_this.open(s.append(".csv"));
      latex_this.open(s.append(".txt"));
      Stats stats_this[FAST_AOM + 1][BF_AOM + 1];

      csv_this << "detect;describe;match;mode;src;ref;matches;inliers;outliers;"
                  "percent;time\n";

      //------------------------------Accumulate stats-------------------------
      for (auto &stats : stats_all) {
        for (int d = Detect::FAST_BEBLID; d < Detect::FAST_AOM; d++) {
          for (int m = Match::FLANN_BEST; m < Match::BF_AOM; m++) {
            for (auto &stat : stats) {
              stats_accumulate(stats_this[d][m], stat[d][m]);
            }
          }
        }
      }

      for (auto &stats : stats_all) {
        for (auto &stat : stats) {
          stats_accumulate(stats_this[FAST_AOM][BF_AOM],
                           stat[FAST_AOM][BF_AOM]);
        }
      }

      //---------------------Normalize and print stats-------------------------
      for (int d = Detect::FAST_BEBLID; d < Detect::FAST_AOM; d++) {
        for (int m = Match::FLANN_BEST; m < Match::BF_AOM; m++) {
          stats_normalize(stats_this[d][m], ITERATIONS * FRAMES);

          #pragma omp critical
          stats_accumulate(stats_global[d][m], stats_this[d][m]);

          print_csv(stats_this[d][m], d, m, csv_this);
          print_latex(stats_this[d][m], d, m, latex_this);
        }
      }

      stats_normalize(stats_this[FAST_AOM][BF_AOM], ITERATIONS * FRAMES);

      #pragma omp critical
      stats_accumulate(stats_global[FAST_AOM][BF_AOM],
                       stats_this[FAST_AOM][BF_AOM]);

      print_csv(stats_this[FAST_AOM][BF_AOM], FAST_AOM, BF_AOM, csv_this);
      print_latex(stats_this[FAST_AOM][BF_AOM], FAST_AOM, BF_AOM, latex_this);

      //------------------------------Close files------------------------------
      csv_this.close();
      latex_this.close();

      auto stop_this = high_resolution_clock::now();

      fprintf(stderr,
              "%d\t|\t%s\t|\t-\t|\t-\t|\t-\t|\t-\t|\t-\t|\t-\t|\t-\t|\t%lu "
              "ms\n",
              omp_get_thread_num(),
              name.substr(0, name.length() < 8 ? name.length() : 8).c_str(),
              duration_cast<milliseconds>(stop_this - start_this).count());
    }

    #pragma omp master
    {
      //-------------------------Initialize global files-----------------------

      fprintf(stderr, "\n--WRITING GLOBAL FILE\n");

      ofstream csv_global;
      ofstream latex_global;
      string s(folder + "global");
      s.append("_f")
          .append(std::to_string(FRAMES))
          .append("_t")
          .append(std::to_string(ITERATIONS))
          .append("_t")
          .append(std::to_string(omp_get_thread_num()));
      csv_global.open(s.append(".csv"));
      latex_global.open(s.append(".txt"));

      csv_global
          << "detect;describe;match;mode;src;ref;matches;inliers;outliers;"
             "percent;time\n";

      int vs = argc - 2;

      //--------------------Normalize and global print stats-------------------
      for (int d = Detect::FAST_BEBLID; d < Detect::FAST_AOM; d++) {
        for (int m = Match::FLANN_BEST; m < Match::BF_AOM; m++) {
          stats_normalize(stats_global[d][m], vs);
          print_csv(stats_global[d][m], d, m, csv_global);
          print_latex(stats_global[d][m], d, m, latex_global);
        }
      }

      stats_normalize(stats_global[FAST_AOM][BF_AOM], vs);
      print_csv(stats_global[FAST_AOM][BF_AOM], FAST_AOM, BF_AOM, csv_global);
      print_latex(stats_global[FAST_AOM][BF_AOM], FAST_AOM, BF_AOM,
                  latex_global);

      //---------------------------Close global files--------------------------
      csv_global.close();
      latex_global.close();

      auto stop_global = high_resolution_clock::now();

      fprintf(stderr, "--FINISHED IN %lu ms\n",
              duration_cast<milliseconds>(stop_global - start_global).count());
    }
  }
}
