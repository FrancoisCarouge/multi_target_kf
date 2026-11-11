// The adaptive acceleration unscented filter in the tracker's model interface. The filter itself is kept as
// is, untyped, until the Kalman library offers an unscented filter: this class only adapts its interface.
#ifndef UKF_MODEL_H
#define UKF_MODEL_H

#include <array>
#include <cmath>
#include <cstddef>

#include "multi_target_kf/typed_models.h"
#include "multi_target_kf/ukf_adaptive_accel.h"

namespace multi_target_kf
{
class UnscentedAcceleration
{
public:
  using filter_type = kf_state;
  static constexpr std::size_t states{9};

  struct snapshot
  {
    double time_stamp;
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
  };

  /// The filter's own configuration, as the original tracker's initialization.
  AdaptiveAccelUKF & filter() { return ukf_; }

  bool output_variances(const std::vector<double> & r_diag) { return ukf_.R(r_diag); }

  [[nodiscard]] filter_type init(const sensor_measurement & z) { return ukf_.initStateFromMeasurements(z); }

  [[nodiscard]] filter_type birth(const sensor_measurement & z, double dt_pred)
  {
    filter_type state{init(z)};
    state.P = ukf_.Q(dt_pred);
    return state;
  }

  bool predict(filter_type & state, double dt)
  {
    state = ukf_.predictX(state, dt);
    return dt > 0;
  }

  void update(filter_type & state, const sensor_measurement & z) { state = ukf_.updateX(z, state); }

  [[nodiscard]] double log_likelihood(const filter_type & state, const sensor_measurement & z)
  {
    return ukf_.logLikelihood(state, z);
  }

  [[nodiscard]] static typed::Volume position_volume(const filter_type & state)
  {
    using mp_units::si::metre;
    return std::sqrt(std::fabs(state.P.block(0, 0, 3, 3).determinant())) * (metre * metre * metre);
  }

  [[nodiscard]] static double state_norm(const filter_type & state) { return state.x.norm(); }

  [[nodiscard]] static bool finite(const filter_type & state) { return !std::isinf(state.x.norm()); }

  [[nodiscard]] static snapshot save(const filter_type & state, double time_stamp)
  {
    return {time_stamp, state.x, state.P};
  }

  static void restore(filter_type & state, const snapshot & s)
  {
    state.x = s.x;
    state.P = s.P;
  }

  [[nodiscard]] static std::array<double, states> state_values(const filter_type & state)
  {
    std::array<double, states> values{};
    for (std::size_t i{0}; i < states; ++i) {
      values[i] = state.x(static_cast<Eigen::Index>(i));
    }
    return values;
  }

  [[nodiscard]] static std::array<double, states * states> covariance_values(const filter_type & state)
  {
    std::array<double, states * states> values{};
    for (std::size_t i{0}; i < states; ++i) {
      for (std::size_t j{0}; j < states; ++j) {
        values[i * states + j] = state.P(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
      }
    }
    return values;
  }

private:
  AdaptiveAccelUKF ukf_;
};
}  // namespace multi_target_kf

#endif  // UKF_MODEL_H
