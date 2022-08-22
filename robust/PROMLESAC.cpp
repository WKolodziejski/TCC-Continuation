#ifndef __RTL_PROMLESAC__
#define __RTL_PROMLESAC__

#include "RANSAC.cpp"
#include <algorithm>
#include <climits>

namespace RTL {

template <class Model, class Datum, class Data>
class PROMLESAC : virtual public RANSAC<Model, Datum, Data> {
 public:
  explicit PROMLESAC(Estimator<Model, Datum, Data> *estimator)
      : RANSAC<Model, Datum, Data>(estimator) {
    //fprintf(stderr, "---PROSAC---\n");
  }

 protected:
  virtual void initialize(const Data &data, int N) {
    RANSAC<Model, Datum, Data>::initialize(data, N);

    for (int i = 0; i < MIN_PTS; i++) {
      t_n *= static_cast<double>(n - i) / (N - i);
    }

    dataError2 = new long double[N];
    assert(dataError2 != nullptr);
    double sigma = INLIER_THRESHOLD / SIGMA_SCALE;
    dataSigma2 = sigma * sigma;
  }

  virtual bool generateModel(const Data &data, Model &model, int M, int N,
                             double *points1, double *points2) {
    int degenerate = 1;
    int num_degenerate_iter = 0;

    while (degenerate) {
      num_degenerate_iter++;

      Datum subset[MIN_PTS];
      sample(data, subset, N);

      double *pts1 = points1;
      double *pts2 = points2;

      for (auto &d : subset) {
        *(pts1++) = d.x;
        *(pts1++) = d.y;
        *(pts2++) = d.rx;
        *(pts2++) = d.ry;
      }

      degenerate = this->toolEstimator->isDegenerate(points1);

      if (num_degenerate_iter > MAX_DEGENERATE_ITER) return false;
    }

    model = this->toolEstimator->computeModel(points1, points2, M);
    return true;
  }

 private:
  virtual void sample(const Data &data, Data subset, int N) {
    int j = 0;
    t++;

    if (t > t_n_prime && n < N) {
      double t_n_plus1 = (t_n * (n + 1.0)) / (n + 1.0 - MIN_PTS);
      t_n_prime += ceil(t_n_plus1 - t_n);
      t_n = t_n_plus1;
      n++;
    }

    std::vector<int> random_numbers;

    if (t > t_n_prime) {
      for (int i = 0; i < MIN_PTS; i++) {
        int rand_number;
        while (std::find(random_numbers.begin(), random_numbers.end(),
                         (rand_number = random(n))) != random_numbers.end()) {
        }

        random_numbers.push_back(rand_number);
        subset[j++] = data[rand_number];
      }
    } else {
      for (int i = 0; i < MIN_PTS - 1; i++) {
        int rand_number;
        while (std::find(random_numbers.begin(), random_numbers.end(),
                         (rand_number = random(n - 1))) !=
               random_numbers.end()) {
        }

        random_numbers.push_back(rand_number);
        subset[j++] = data[rand_number];
      }

      subset[j++] = data[n];
    }
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

  int random(int N) {
    int index;
    this->get_rand_indices(N, 1, &index, &this->seed);
    return index;
  }

  double t_n = MIN_TRIALS;
  int n = MIN_PTS;
  int t = 0;
  double t_n_prime = 1.0;
  long double *dataError2;
  long double dataSigma2;
};

}  // namespace RTL

#endif  // End of '__RTL_PROMLESAC__'
