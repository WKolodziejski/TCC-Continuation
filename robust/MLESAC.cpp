#ifndef __RTL_MLESAC__
#define __RTL_MLESAC__

#include "MSAC.cpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace RTL {

template <class Model, class Datum, class Data>
class MLESAC : virtual public RANSAC<Model, Datum, Data> {
 public:
  explicit MLESAC(Estimator<Model, Datum, Data> *estimator)
      : RANSAC<Model, Datum, Data>(estimator) {
    // dataError2 = nullptr;
    // fprintf(stderr, "---MLESAC---\n");
  }

 protected:
  virtual void initialize(const Data &data, int N) {
    RANSAC<Model, Datum, Data>::initialize(data, N);
    dataError2 = new long double[N];
    assert(dataError2 != nullptr);
    double sigma = INLIER_THRESHOLD / SIGMA_SCALE;
    dataSigma2 = sigma * sigma;
  }

  virtual Error evaluateModel(const Model &model, const Data &data, int N) {
    // Calculate squared errors
    double minError = HUGE_VAL, maxError = -HUGE_VAL;

    double sum_distance = 0;
    double sum_distance_squared = 0;
    int num_inliers = 0;
    double variance = HUGE_VAL;

    for (int i = 0; i < N; i++) {
      double error = this->toolEstimator->computeError(model, data[i]);
      if (error < minError) minError = error;
      if (error > maxError) maxError = error;
      dataError2[i] = error * error;

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

    // Estimate the inlier ratio using EM
    const double nu = maxError - minError;
    long double gamma = 0.5;
    for (int iter = 0; iter < ITERATION; iter++) {
      long double sumPosteriorProb = 0;
      const long double probOutlier = (1 - gamma) / nu;
      const long double probInlierCoeff = gamma / sqrt(2 * M_PI * dataSigma2);
      for (int i = 0; i < N; i++) {
        long double de2 = dataError2[i];
        long double x = -0.5 * de2 / dataSigma2;
        long double e = exp(x);
        long double probInlier = probInlierCoeff * e;
        sumPosteriorProb += probInlier / (probInlier + probOutlier);
      }
      gamma = sumPosteriorProb / N;
    }

    // Evaluate the model
    long double sumLogLikelihood = 0;
    const long double probOutlier = (1 - gamma) / nu;
    const long double probInlierCoeff = gamma / sqrt(2 * M_PI * dataSigma2);
    for (int i = 0; i < N; i++) {
      long double de2 = dataError2[i];
      long double x = -0.5 * de2 / dataSigma2;
      long double e = exp(x);
      long double probInlier = probInlierCoeff * e;
      //      double probInlier =
      //          probInlierCoeff * exp(-0.5 * dataError2[i] / dataSigma2);
      sumLogLikelihood = sumLogLikelihood - log(probInlier + probOutlier);
    }
    Error error{ static_cast<double>(sumLogLikelihood), variance };
    return error;
  }

  virtual void terminate(const Data &data, int N, const Model &bestModel) {
    if (dataError2 != nullptr) {
      delete[] dataError2;
      dataError2 = nullptr;
    }
    RANSAC<Model, Datum, Data>::terminate(bestModel, data, N);
  }

  long double *dataError2;

  long double dataSigma2;
};

}  // namespace RTL

#endif  // End of '__RTL_MLESAC__'
