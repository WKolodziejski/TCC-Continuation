#ifndef __RTL_RANSAC__
#define __RTL_RANSAC__

#include "Base.cpp"
#include <cmath>
#include <cassert>
#include <cstdio>

namespace RTL {

struct Error {
  double loss;
  double variance;
};

template <class Model, class Datum, class Data>
class RANSAC {
 public:
  explicit RANSAC(Estimator<Model, Datum, Data> *estimator) {
    assert(estimator != NULL);
    toolEstimator = estimator;
  }

  virtual void compute(const Data &data, int N, MotionModel *params_by_motion) {
    double *points1 = (double *)malloc(sizeof(*points1) * N * 2);
    double *points2 = (double *)malloc(sizeof(*points2) * N * 2);

    RTL::Affine model;

    findBest(model, data, N, MIN_PTS, points1, points2);
    findInliers(model, data, N, params_by_motion);

    if (params_by_motion->num_inliers >= 3) {
      for (int i = 0; i < params_by_motion->num_inliers; ++i) {
        const int index = params_by_motion->inliers[i];
        points1[i * 2] = data[index].x;
        points1[i * 2 + 1] = data[index].y;
        points2[i * 2] = data[index].rx;
        points2[i * 2 + 1] = data[index].ry;
      }

      model = toolEstimator->computeModel(points1, points2,
                                          params_by_motion->num_inliers);

      findInliers(model, data, N, params_by_motion);
    }

    for (int i = 0; i < MAX_PARAMDIM - 1; ++i) {
      params_by_motion->params[i] = model.mat[i];
    }

    free(points1);
    free(points2);
  }

 protected:
  virtual void findBest(Model &best, const Data &data, int N, int M,
                        double *points1, double *points2) {
    assert(N > 0 && M > 0);

    initialize(data, N);

    double bestloss = HUGE_VAL;
    double bestvariance = HUGE_VAL;
    int iteration = 0;

    while (iteration < MIN_TRIALS) {
      iteration++;

      Model model;

      if (!generateModel(data, model, M, N, points1, points2)) continue;

      if (!toolEstimator->fitsLSE(model)) continue;

      Error error = evaluateModel(model, data, N);

      if (error.loss < bestloss) {
        if (error.variance < bestvariance) {
          best = model;
          bestloss = error.loss;
          bestvariance = error.variance;
        }
      }
    }

    terminate(best, data, N);
  }

  virtual void findInliers(const Model &model, const Data &data, int N,
                           MotionModel *params_by_motion) {
    params_by_motion->num_inliers = 0;

    for (int i = 0; i < N; i++) {
      double error = toolEstimator->computeError(model, data[i]);

      if (fabs(error) < INLIER_THRESHOLD) {
        params_by_motion->inliers[params_by_motion->num_inliers++] = i;
      }
    }
  }

  virtual bool generateModel(const Data &data, Model &model, int M, int N,
                             double *points1, double *points2) {
    int indices[MIN_PTS];

    int degenerate = 1;
    int num_degenerate_iter = 0;

    while (degenerate) {
      num_degenerate_iter++;

      if (!get_rand_indices(N, M, indices, &seed)) return false;

      double *pts1 = points1;
      double *pts2 = points2;

      for (int i : indices) {
        *(pts1++) = data[i].x;
        *(pts1++) = data[i].y;
        *(pts2++) = data[i].rx;
        *(pts2++) = data[i].ry;
      }

      degenerate = toolEstimator->isDegenerate(points1);

      if (num_degenerate_iter > MAX_DEGENERATE_ITER) return false;
    }

    model = toolEstimator->computeModel(points1, points2, M);
    return true;
  }

  virtual Error evaluateModel(const Model &model, const Data &data, int N) {
    double loss = 0;
    double sum_distance = 0;
    double sum_distance_squared = 0;
    int num_inliers = 0;
    double variance = HUGE_VAL;

    for (int i = 0; i < N; i++) {
      double error = toolEstimator->computeError(model, data[i]);
      loss += (fabs(error) >= INLIER_THRESHOLD);

      if (error < INLIER_THRESHOLD) {
        num_inliers++;
        sum_distance += error;
        sum_distance_squared += error * error;
      }
    }

    if (num_inliers > 1) {
      double mean_distance = sum_distance / ((double)num_inliers);
      variance = sum_distance_squared / ((double)num_inliers - 1.0) -
                 mean_distance * mean_distance * ((double)num_inliers) /
                     ((double)num_inliers - 1.0);
    }

    Error error{ loss, variance };
    return error;
  }

  virtual void initialize(const Data &data, int N) {
    (void)data;
    seed = (unsigned int)N;
  }

  virtual void terminate(const Model &bestModel, const Data &data, int N) {
    (void)bestModel;
    (void)data;
    (void)N;
  }

  static int get_rand_indices(int npoints, int minpts, int *indices,
                              unsigned int *seed) {
    int i, j;
    int ptr = lcg_rand16(seed) % npoints;
    if (minpts > npoints) return 0;
    indices[0] = ptr;
    ptr = (ptr == npoints - 1 ? 0 : ptr + 1);
    i = 1;
    while (i < minpts) {
      int index = lcg_rand16(seed) % npoints;
      while (index) {
        ptr = (ptr == npoints - 1 ? 0 : ptr + 1);
        for (j = 0; j < i; ++j) {
          if (indices[j] == ptr) break;
        }
        if (j == i) index--;
      }
      indices[i++] = ptr;
    }
    return 1;
  }

  static inline unsigned int lcg_rand16(unsigned int *state) {
    *state = (unsigned int)(*state * 1103515245ULL + 12345);
    return *state / 65536 % 32768;
  }

  static void copy_points_at_indices(double *dest, const double *src,
                                     const int *indices, int num_points) {
    for (int i = 0; i < num_points; ++i) {
      const int index = indices[i];
      dest[i * 2] = src[index * 2];
      dest[i * 2 + 1] = src[index * 2 + 1];
    }
  }

  Estimator<Model, Datum, Data> *toolEstimator;

  // int paramIteration;

  // double paramThreshold;

  unsigned int seed;
};  // End of 'RANSAC'

}  // namespace RTL

#endif  // End of '__RTL_RANSAC__'
