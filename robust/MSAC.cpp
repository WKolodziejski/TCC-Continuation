#ifndef __RTL_MSAC__
#define __RTL_MSAC__

#include "RANSAC.cpp"

namespace RTL {

template <class Model, class Datum, class Data>
class MSAC : virtual public RANSAC<Model, Datum, Data> {
 public:
  explicit MSAC(Estimator<Model, Datum, Data> *estimator)
      : RANSAC<Model, Datum, Data>(estimator) {
  }

 protected:
  virtual Error evaluateModel(const Model &model, const Data &data, int N) {
    double loss = 0;
    double sum_distance = 0;
    double sum_distance_squared = 0;
    int num_inliers = 0;
    double variance = HUGE_VAL;

    for (int i = 0; i < N; i++) {
      double error = this->toolEstimator->computeError(model, data[i]);
      if (error > INLIER_THRESHOLD || error < -INLIER_THRESHOLD)
        loss += INLIER_THRESHOLD * INLIER_THRESHOLD;
      else
        loss += error * error;

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
};

}  // namespace RTL

#endif  // End of '__RTL_MSAC__'
