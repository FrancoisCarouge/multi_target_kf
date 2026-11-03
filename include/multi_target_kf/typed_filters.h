// The constant velocity and constant acceleration filters on the FrancoisCarouge Kalman library, with
// physical units on every element. Defined in src/typed_filters.cpp, the only translation unit of this package
// that needs C++26 and the library's headers: this header only names Eigen.
#ifndef TYPED_FILTERS_H
#define TYPED_FILTERS_H

#include <Eigen/Dense>

namespace multi_target_kf {

// The prediction over dt [s] of a state and covariance, under the spectral density of the white noise driving
// the model: spectral_acceleration [m/s^1.5] squared, or spectral_jerk [m/s^2.5] squared.
void predictConstantVelocity(Eigen::VectorXd& x, Eigen::MatrixXd& P, double dt, double spectral_acceleration);
void predictConstantAcceleration(Eigen::VectorXd& x, Eigen::MatrixXd& P, double dt, double spectral_jerk);

// The correction of a state and covariance by a position measurement z [m] of covariance R [m^2].
void updateConstantVelocity(Eigen::VectorXd& x, Eigen::MatrixXd& P, const Eigen::VectorXd& z, const Eigen::MatrixXd& R);
void updateConstantAcceleration(Eigen::VectorXd& x, Eigen::MatrixXd& P, const Eigen::VectorXd& z, const Eigen::MatrixXd& R);

} // namespace multi_target_kf

#endif // TYPED_FILTERS_H
