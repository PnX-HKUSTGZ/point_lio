#include "IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

typedef MTK::vect<24, double> vector24;
MTK_BUILD_MANIFOLD(linear_state, ((vector24, values)));
using Filter = esekfom::esekf<linear_state, 24>;
using Matrix24 = Eigen::Matrix<double, 24, 24>;
using Vector24 = Eigen::Matrix<double, 24, 1>;

static Eigen::MatrixXd measurement_h;
static Eigen::VectorXd measurement_z;
static double measurement_noise;

Vector24 zero_process(linear_state &, const linear_state &) {return Vector24::Zero();}
Matrix24 zero_jacobian(linear_state &, const linear_state &) {return Matrix24::Zero();}
void linear_measurement(linear_state &, Eigen::Matrix3d, Eigen::Matrix3d,
                        esekfom::dyn_share_modified<double> & data)
{
  data.valid = true;
  data.h_x = measurement_h;
  data.z = measurement_z;
  data.M_Noise = measurement_noise;
}

void require(bool ok, const char * message)
{
  if (!ok) {std::cerr << message << '\n'; std::exit(1);}
}

struct Result {Vector24 state; Matrix24 covariance;};
Result run(int count, double noise, bool repeated)
{
  measurement_noise = noise;
  measurement_h = Eigen::MatrixXd::Zero(count, 12);
  measurement_z = Eigen::VectorXd::Zero(count);
  for (int i = 0; i < count; ++i) {
    if (repeated) {
      measurement_h(i, 0) = 1.0;
      measurement_z(i) = 2.0;
    } else {
      for (int j = 0; j < 12; ++j) {
        measurement_h(i, j) = std::sin((i + 1) * (j + 2) * 0.13) / 4.0;
      }
      measurement_z(i) = std::cos((i + 1) * 0.17);
    }
  }
  Matrix24 factor = Matrix24::Identity();
  factor(14, 0) = .23;
  factor(18, 4) = -.16;
  const Matrix24 prior = factor * factor.transpose();
  linear_state initial;
  initial.values.setZero();
  Filter filter(initial, prior);
  filter.init_dyn_share_modified_2h(zero_process, zero_jacobian, linear_measurement);
  require(filter.update_iterated_dyn_share_modified(), "EKF update failed");
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(count, 24);
  H.leftCols(12) = measurement_h;
  const Eigen::MatrixXd S = H * prior * H.transpose() + noise * Eigen::MatrixXd::Identity(count, count);
  const Eigen::MatrixXd gain = prior * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(count, count));
  const Vector24 oracle_state = gain * measurement_z;
  const Matrix24 oracle_cov = prior - gain * H * prior;
  Result result{filter.x_.values, filter.P_};
  require((result.state-oracle_state).norm() < 2e-8, "state differs from direct Kalman oracle");
  require((result.covariance-oracle_cov).norm() < 2e-8, "covariance differs from direct Kalman oracle");
  return result;
}

int main()
{
  for (double noise : {0.001, 0.1, 2.0}) {
    for (int count : {1, 23, 24, 25, 64}) {
      run(count, noise, false);
      run(count, noise, true);
    }
  }
  const auto few = run(1, .1, true);
  const auto many = run(64, .1, true);
  require(many.covariance(0,0) < few.covariance(0,0), "repeated observations must increase precision");
  require(std::abs(2.0-many.state(0)) < std::abs(2.0-few.state(0)), "repeated observations must improve estimate");
  // Equal N/R across the branch boundary yields the same information and update.
  const auto below = run(23, .23, true);
  const auto above = run(24, .24, true);
  require((below.state-above.state).norm() < 1e-8, "branch boundary state mismatch");
  require((below.covariance-above.covariance).norm() < 1e-8, "branch boundary covariance mismatch");
  return 0;
}
