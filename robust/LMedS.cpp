#ifndef __RTL_LMEDS__
#define __RTL_LMEDS__

#include "RANSAC.cpp"
#include <algorithm>

namespace RTL {

template <class Model, class Datum, class Data>
class LMedS : virtual public RANSAC<Model, Datum, Data> {
 public:
  explicit LMedS(Estimator<Model, Datum, Data> *estimator)
      : RANSAC<Model, Datum, Data>(estimator) {
  }

 protected:
  virtual Error evaluateModel(const Model &model, const Data &data, int N) {
    std::vector<double> errors(N);

    double sum_distance = 0;
    double sum_distance_squared = 0;
    int num_inliers = 0;
    double variance = HUGE_VAL;

    for (int i = 0; i < N; i++) {
      errors[i] = fabs(this->toolEstimator->computeError(model, data[i]));
      double error = errors[i];

      if (error < INLIER_THRESHOLD) {
        num_inliers++;
        sum_distance += error;
        sum_distance_squared += error * error;
      }
    }

    std::nth_element(errors.begin(), errors.begin() + N / 2, errors.end());

    if (num_inliers > 1) {
      double mean_distance = sum_distance / ((double)num_inliers);
      variance = sum_distance_squared / ((double)num_inliers - 1.0) -
                 mean_distance * mean_distance * ((double)num_inliers) /
                     ((double)num_inliers - 1.0);
    }

    Error error{ errors[N / 2], variance };
    return error;
  }
};

}  // namespace RTL

#endif  // End of '__RTL_LMEDS__'
