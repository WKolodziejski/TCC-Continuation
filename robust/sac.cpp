#ifndef __SAC__
#define __SAC__

#include <stdio.h>
#include "sac.hpp"
#include "../av1/ransac.c"
#include "../opencv/draw.hpp"
#include "../utils.hpp"
#include <cmath>

RTL::Estimator<RTL::Affine, Correspondence, Correspondence *> *getEstimator(
    TransformationType type) {
  switch (type) {
    case AFFINE: return new AffineEstimator();
    case ROTZOOM: return new RotZoomEstimator();
    case TRANSLATION: return new TranslationEstimator();
    default: assert(0);
  }
}

double segmented_error(const Mat &src_img, const Mat &ref_img,
                       const double mat[8], int x, int y, MatrixMap **map, int k) {
  Mat warp_mat = Mat::zeros(2, 3, CV_64FC1);
  warp_mat.at<double>(0) = mat[2];  // scale x
  warp_mat.at<double>(1) = mat[3];  // rot +
  warp_mat.at<double>(2) = mat[0];  // trans x
  warp_mat.at<double>(3) = mat[4];  // rot -
  warp_mat.at<double>(4) = mat[5];  // scale y
  warp_mat.at<double>(5) = mat[1];  // trans y

  Mat warped_img;
  warpAffine(src_img, warped_img, warp_mat, src_img.size(), INTER_AREA);

  Mat error_img;
  subtract(warped_img, ref_img, error_img);
  multiply(error_img, error_img, error_img);

  for (int xi = 0; xi < x; xi++) {
    for (int yj = 0; yj < y; yj++) {
      Rect roi = Rect(xi * 32, yj * 32, 32, 32);
      Mat cropped = error_img(roi);

      double error = sum(cropped)[0];

      if (error < map[xi][yj].error) {
        map[xi][yj].error = error;
        map[xi][yj].k = k;

        // Copia a matriz
        for (int j = 0; j < 8; j++) {
          map[xi][yj].mat[j] = mat[j];
        }
      }
    }
  }
}

// TODO: também comparar com versão padrão
void estimate_clustered(Mat &src_img, Mat &ref_img,
                        Correspondence *correspondences,
                        int num_correspondences,
                        TransformationType transformation_type, Estimate type,
                        Stats &stats, const string &name, int frame) {
  //    draw_clustered_motion_field(src_img, ref_img, correspondences,
  //                                num_correspondences,
  //                                formatName("tcc_clusters", frame));

  Mat labels;
  Mat centers;
  Mat samples(num_correspondences, 1, CV_32F);

  // Calcula distâncias dos MVs nas correspondências
  for (int i = 0; i < num_correspondences; i++) {
    samples.at<float>(i) = distance(correspondences[i]);
  }

  double last_error = 0;
  int k = 1;
  int a = 0;

  // Escolhe o melhor valor de K usando a regra do joelho
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

  int x = floor(src_img.size().width / 32);
  int y = floor(src_img.size().height / 32);

//  MatrixMap &map = (MatrixMap*) malloc(x * sizeof(MatrixMap *));
//  for (int ww = 0; ww < y; ww++) {
//      &map[ww] = malloc(y * sizeof(MatrixMap));
//  }

  MatrixMap **map = new MatrixMap*[x];
  for (int xi = 0; xi < x; xi++) {
    map[xi] = new MatrixMap[y];
  }

  labels.release();
  centers.release();

  fprintf(stderr, "k = %d\n", k);

  kmeans(samples, k, labels, TermCriteria(3, 10000, 0.0001), 5,
         KMEANS_PP_CENTERS, centers);

  vector<Correspondence> clusters[k];

  // Popula os clusters com as respectivas correspondências
  for (int i = 0; i < num_correspondences; i++) {
    Correspondence c = correspondences[i];
    clusters[labels.at<int>(i)].push_back(c);
  }

  double best_mat[8];
  double best_error = HUGE_VAL;
  int best_k = 0;

  // Escolhe a melhor matriz afim, dentre as K geradas
  for (int i = 0; i < k; i++) {
    fprintf(stderr, "----------%d----------\n", i);

    double mat[8];
    size_t size = clusters[i].size();

    if (size < 3) {
      fprintf(stderr, "SKIP size < 3\n");
      continue;
    }

    bool identity = true;
    for (double j : mat) {
      if (j != 0) {
        identity = false;
        break;
      }
    }

    if (identity) {
      fprintf(stderr, "SKIP identity\n");
      continue;
    }

    // Estima quantidade de inliers para o cluster atual
    double inliers = estimate(clusters[i].data(), size, transformation_type,
                              type, stats, mat);

    if (inliers < 0.01) {
      fprintf(stderr, "SKIP inliers < 0.01\n");
      continue;
    }

    //    double error = draw_warped(src_img, ref_img, mat,
    //                               formatNameCluster("warped", name, frame,
    //                               i));

    //    draw_motion_field(src_img, ref_img, clusters[i].data(), size,
    //                      formatNameCluster("motion_field", name, frame, i));

    segmented_error(src_img, ref_img, mat, x, y, map, i);

//    if (error < best_error) {
//      best_error = error;
//      best_k = i;
//      for (int j = 0; j < 8; j++) {
//        best_mat[j] = mat[j];
//      }
//    }
  }

  draw_k_warped_image(src_img, x, y, map, name, frame);

  fprintf(stderr, "----------BEST---------\n");

  draw_motion_field(
      src_img, ref_img, clusters[best_k].data(), clusters[best_k].size(),
      formatNameCluster("BEST_motion_field", name, frame, best_k));

  draw_warped(src_img, ref_img, best_mat,
              formatNameCluster("BEST_warped", name, frame, best_k));
}

double estimate(Correspondence *correspondences, int num_correspondences,
                TransformationType transformation_type, Estimate type,
                Stats &stats, double mat[8]) {
  RTL::Estimator<RTL::Affine, Correspondence, Correspondence *> *estimator =
      getEstimator(transformation_type);
  MotionModel params_by_motion[1];
  params_by_motion[0].num_inliers = 0;
  params_by_motion[0].inliers = static_cast<int *>(
      malloc(sizeof(*(params_by_motion[0].inliers)) * 2 * num_correspondences));

  if (type == Estimate::RANSAC) {
    RTL::RANSAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::MLESAC) {
    RTL::MLESAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::MSAC) {
    RTL::MSAC<RTL::Affine, Correspondence, Correspondence *> ransac(estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::PROSAC) {
    RTL::PROSAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::LMEDS) {
    RTL::LMedS<RTL::Affine, Correspondence, Correspondence *> ransac(estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::PROMSAC) {
    RTL::PROMSAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::PROMLESAC) {
    RTL::PROMLESAC<RTL::Affine, Correspondence, Correspondence *> ransac(
        estimator);
    ransac.compute(correspondences, num_correspondences, params_by_motion);

  } else if (type == Estimate::RANSAC_AOM) {
    int num_inliers_by_motion[1];
    RansacFunc ransac = av1_get_ransac_type(transformation_type);
    ransac((int *)correspondences, num_correspondences, num_inliers_by_motion,
           params_by_motion, 1);

  } else {
    assert(0);
  }

  stats.outliers_num = num_correspondences - params_by_motion->num_inliers;
  stats.inliers_num = params_by_motion->num_inliers;
  stats.inliers_per =
      stats.matches_num == 0
          ? 0.0
          : (float)stats.inliers_num / (float)stats.matches_num * 100;

  fprintf(stderr, "i = %f\n", stats.inliers_per);

  for (int i = 0; i < 8; i++) {
    mat[i] = params_by_motion->params[i];
  }

  free(params_by_motion[0].inliers);

  delete estimator;

  return stats.inliers_per;
}

// void estimate_av1(int *correspondences, int num_correspondences,
//                   TransformationType transformation_type, Stats &stats) {
//   int num_inliers_by_motion[1];
//   MotionModel params_by_motion[1];
//   params_by_motion[0].num_inliers = 0;
//   params_by_motion[0].inliers = static_cast<int *>(
//       malloc(sizeof(*(params_by_motion[0].inliers)) * 2 *
//       num_correspondences));
//
//   RansacFunc ransac = av1_get_ransac_type(transformation_type);
//   ransac(correspondences, num_correspondences, num_inliers_by_motion,
//          params_by_motion, 1);
//
//   stats.outliers_num = num_correspondences - num_inliers_by_motion[0];
//   stats.inliers_num = num_inliers_by_motion[0];
//   stats.inliers_per =
//       stats.matches_num == 0
//           ? 0.0
//           : (float)stats.inliers_num / (float)stats.matches_num * 100;
//
//   // Pontos: inliers / outliers
//   //  fprintf(stderr, "%d / %d : %f\n", params_by_motion->num_inliers,
//   outliers,
//   //  p);
//
//   free(params_by_motion[0].inliers);
// }

#endif  // End of '__SAC__'