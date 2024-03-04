#include "draw.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include "../utils.hpp"
#include "../av1/warp_affine.c"

#define DRAW true

Scalar get_cluster_color(int k) {
  switch (k) {
    case 0: return Scalar(255, 0, 0);
    case 1: return Scalar(0, 255, 0);
    case 2: return Scalar(127, 0, 127);
    case 3: return Scalar(0, 255, 255);
    case 4: return Scalar(255, 0, 255);
    case 5: return Scalar(255, 255, 0);
    case 6: return Scalar(255, 127, 0);
    case 7: return Scalar(0, 255, 127);
    case 8: return Scalar(127, 0, 255);
    case 9: return Scalar(127, 127, 255);
    case 10: return Scalar(255, 127, 127);
    default: return Scalar(255, 255, 255);
  }
}

Rect cut_rect(int xi, int yj) {
  return { xi * WARP_BLOCK_SIZE, yj * WARP_BLOCK_SIZE, WARP_BLOCK_SIZE,
           WARP_BLOCK_SIZE };
}

Mat cut_block(const Mat &src_img, int xi, int yj) {
  // Rect que recorta a imagem
  Rect rect = cut_rect(xi, yj);

  // Imagem preta
  Mat block_img = Mat::zeros(src_img.rows, src_img.cols, src_img.type());
  // Bloco recortado
  Mat block = src_img(rect);
  // Copia apenas o bloco para a imagem preta
  block.copyTo(block_img(rect));

  return block_img;
}

Mat cut_overlay(const Mat &src_img, int xi, int yj, const MatrixMap &map) {
  // Rect que recorta a imagem
  Rect rect = Rect(xi * WARP_BLOCK_SIZE, yj * WARP_BLOCK_SIZE, WARP_BLOCK_SIZE,
                   WARP_BLOCK_SIZE);

  Mat color =
      Mat(src_img.rows, src_img.cols, CV_8UC3,
          map.zero_motion ? Scalar(0, 0, 255) : get_cluster_color(map.k));

  // Imagem preta
  Mat block_img = Mat::zeros(src_img.rows, src_img.cols, CV_8UC3);
  // Bloco recortado
  Mat block = color(rect);
  // Copia apenas o bloco para a imagem preta
  block.copyTo(block_img(rect));

  return block_img;
}

void draw_k_warped_image(const Mat &src_img, const Mat &ref_img, int x, int y,
                         MatrixMap **map, const string &inv_name,
                         const string &fwd_name) {
  if (!DRAW) return;

  Mat inv_warped_img = Mat::zeros(src_img.rows, src_img.cols, src_img.type());
  Mat fwd_warped_img = Mat::zeros(src_img.rows, src_img.cols, src_img.type());
  Mat inv_clusters_img = Mat::zeros(src_img.rows, src_img.cols, CV_8UC3);
  Mat fwd_clusters_img = Mat::zeros(src_img.rows, src_img.cols, CV_8UC3);

  for (int xi = 0; xi < x; xi++) {
    for (int yj = 0; yj < y; yj++) {
      Mat warp_mat = parse_affine_mat(map[xi][yj].mat);

      // ---------------TRANSFORMA ANTES E CORTA NO DESTINO---------------------
      //      {
      //        float sx = (xi * WARP_BLOCK_SIZE) + (WARP_BLOCK_SIZE / 2);
      //        float sy = (yj * WARP_BLOCK_SIZE) + (WARP_BLOCK_SIZE / 2);
      //
      //        double px = (warp_mat.at<double>(0) * sx +
      //        warp_mat.at<double>(1) * sy +
      //                    warp_mat.at<double>(2));
      //
      //        double py = (warp_mat.at<double>(3) * sx +
      //        warp_mat.at<double>(4) * sy +
      //                    warp_mat.at<double>(5));
      //
      //        int rx = px / WARP_BLOCK_SIZE;
      //        int ry = py / WARP_BLOCK_SIZE;
      //
      //        if (!(rx > y || ry > y || rx < 0 || ry < 0)) {
      //          Mat warped;
      //          warpAffine(src_img, warped, warp_mat, src_img.size());
      //
      //          Mat block = cut_block(warped, rx, ry);
      //          Mat overlay = cut_overlay(warped, rx, ry, map[xi][yj].k);
      //
      //          add(block, fwd_warped_img, fwd_warped_img);
      //          add(overlay, fwd_clusters_img, fwd_clusters_img);
      //        }
      //      }

      // ------------------CORTA ANTES E TRANSFORMA DEPOIS----------------------
      {
        Mat block = cut_block(src_img, xi, yj);
        Mat overlay = cut_overlay(src_img, xi, yj, map[xi][yj]);

        Mat warped_block;
        warpAffine(block, warped_block, warp_mat, block.size(), INTER_AREA);
        add(warped_block, fwd_warped_img, fwd_warped_img);

        Mat warped_overlay;
        warpAffine(overlay, warped_overlay, warp_mat, block.size(), INTER_AREA);
        add(warped_overlay, fwd_clusters_img, fwd_clusters_img);
      }

      // ------------------TRANSFORMA ANTES E CORTA DEPOIS----------------------
      {
        Mat warped;
        warpAffine(src_img, warped, warp_mat, src_img.size());

        Mat block = cut_block(warped, xi, yj);
        Mat overlay = cut_overlay(warped, xi, yj, map[xi][yj]);

        add(block, inv_warped_img, inv_warped_img);
        add(overlay, inv_clusters_img, inv_clusters_img);
      }
    }
  }

  // Para mostrar imagem de ref ao fundo
  addWeighted(ref_img, 0.5, inv_warped_img, 0.5, 0, inv_warped_img);
  addWeighted(ref_img, 0.5, fwd_warped_img, 0.5, 0, fwd_warped_img);

  Mat inv_warped_rgb;
  cv::cvtColor(inv_warped_img, inv_warped_rgb, cv::COLOR_GRAY2RGB);

  Mat fwd_warped_rgb;
  cv::cvtColor(fwd_warped_img, fwd_warped_rgb, cv::COLOR_GRAY2RGB);

  Mat inv_map_img;
  addWeighted(inv_warped_rgb, 1, inv_clusters_img, 0.5, 0, inv_map_img);

  Mat fwd_map_img;
  addWeighted(fwd_warped_rgb, 1, fwd_clusters_img, 0.5, 0, fwd_map_img);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(inv_name, inv_map_img, params);
  imwrite(fwd_name, fwd_map_img, params);
}

void draw_matches(Mat &src_img, Mat &ref_img,
                  std::vector<KeyPoint> &src_keypoints,
                  std::vector<KeyPoint> &ref_keypoints,
                  std::vector<DMatch> &good_matches, const string &name) {
//  if (!DRAW) return;

  Scalar color_lines = Scalar(0, 255, 0);
  Scalar color_keypoints = Scalar(255, 0, 0);

  Mat img;
  drawMatches(src_img, src_keypoints, ref_img, ref_keypoints, good_matches, img,
              color_lines, color_keypoints, std::vector<char>(),
              DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);
}

void draw_motion_field(Mat &src_img, Mat &ref_img,
                       Correspondence *correspondences, int num_correspondences,
                       const string &name) {
  if (!DRAW) return;

  Mat src;
  Mat ref;
  Mat mask = Mat::zeros(src_img.size(), CV_8UC3);
  cv::cvtColor(src_img, src, cv::COLOR_GRAY2RGB);
  cv::cvtColor(ref_img, ref, cv::COLOR_GRAY2RGB);

  for (int i = 0; i < num_correspondences; i++) {
    Correspondence c = correspondences[i];
    double d = distance(c);

    if (d < 1) {
      circle(mask, Point2f(c.x, c.y), 5, Scalar(0, 0, 255), 1, LINE_AA);
    } else {
      arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                  Scalar(255 - d * 5, d * 5, angle(c)), 1, LINE_AA);
    }
  }

  Mat img;
  addWeighted(src, 0.5, ref, 0.5, 0, img);
  add(img, mask, img);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);
}

void draw_clustered_motion_field(Mat &src_img, Mat &ref_img,
                                 Correspondence *correspondences,
                                 int num_correspondences, const string &name) {
  if (!DRAW) return;

  Mat src;
  Mat ref;
  Mat mask = Mat::zeros(src_img.size(), CV_8UC3);
  cv::cvtColor(src_img, src, cv::COLOR_GRAY2RGB);
  cv::cvtColor(ref_img, ref, cv::COLOR_GRAY2RGB);

  Mat labels;
  Mat centers;
  Mat samples(num_correspondences, 1, CV_32F);

  for (int i = 0; i < num_correspondences; i++) {
    samples.at<float>(i) = distance(correspondences[i]);
  }

  double last_error = 0;
  int k = 1;
  int a = 0;
  for (int i = k; i < 10; i++) {
    double error = kmeans(samples, i, labels, TermCriteria(3, 10000, 0.0001), 5,
                          KMEANS_PP_CENTERS, centers);

    if (error < last_error / 2) {
      k = i;
      a++;
    }

    // if attempts > 2 stop increasing k
    if (a > 1) {
      break;
    }

    last_error = error;
  }

  labels.release();
  centers.release();

  kmeans(samples, k, labels, TermCriteria(3, 10000, 0.0001), 5,
         KMEANS_PP_CENTERS, centers);

  vector<Correspondence> clusters[k];

  for (int i = 0; i < num_correspondences; i++) {
    Correspondence c = correspondences[i];
    clusters[labels.at<int>(i)].push_back(c);
  }

  for (int i = 0; i < num_correspondences; i++) {
    Correspondence c = correspondences[i];

    if (distance(c) < 1) {
      circle(mask, Point2f(c.x, c.y), 5, Scalar(0, 0, 255), 1, LINE_AA);
      continue;
    }

    if (clusters[labels.at<int>(i)].size() < 3) {
      arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                  Scalar(0, 0, 255), 1, LINE_AA);
      continue;
    }

    switch (labels.at<int>(i)) {
      case 0:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 0, 0), 1, LINE_AA);
        break;

      case 1:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(0, 255, 0), 1, LINE_AA);
        break;

      case 2:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(127, 0, 127), 1, LINE_AA);
        break;

      case 3:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(0, 255, 255), 1, LINE_AA);
        break;

      case 4:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 0, 255), 1, LINE_AA);
        break;

      case 5:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 255, 0), 1, LINE_AA);
        break;

      case 6:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 127, 0), 1, LINE_AA);
        break;

      case 7:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(0, 255, 127), 1, LINE_AA);
        break;

      case 8:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(127, 0, 255), 1, LINE_AA);
        break;

      case 9:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(127, 127, 255), 1, LINE_AA);
        break;

      case 10:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 127, 127), 1, LINE_AA);
        break;

      default:
        arrowedLine(mask, Point2f(c.x, c.y), Point2f(c.rx, c.ry),
                    Scalar(255, 255, 255), 1, LINE_AA);
    }
  }

  Mat img;
  addWeighted(src, 0.5, ref, 0.5, 0, img);
  add(img, mask, img);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);
}

void draw_warped(const Mat &src_img, const Mat &ref_img, const double mat[8],
                   const string &name) {
  if (!DRAW) return;

  Mat warp_mat = parse_affine_mat(mat);

  Mat img;
  warpAffine(src_img, img, warp_mat, src_img.size(), INTER_AREA);

  addWeighted(img, 0.5, ref_img, 0.5, 0, img);

  Mat error_img;
  subtract(img, ref_img, error_img);

  multiply(error_img, error_img, error_img);

  vector<int> params;
  params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  params.push_back(0);
  imwrite(name, img, params);
}