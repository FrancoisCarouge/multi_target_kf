// The constant velocity and constant acceleration motion models as typed Kalman filters: every state,
// covariance, transition, and noise element carries its physical unit, over SparseLinearAlgebra storage. Each
// track owns its filter. The arithmetic follows the original models' order of operations.
#ifndef TYPED_MODELS_H
#define TYPED_MODELS_H

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include "multi_target_kf/dt_warning.h"
#include "multi_target_kf/structs.h"
#include "multi_target_kf/tracker_config.h"
#include "multi_target_kf/typed_linear_algebra.h"

namespace multi_target_kf::typed
{
namespace kf = fcarouge::kalman_filter::internal;

using Volume = Quantity<mp_units::si::metre * mp_units::si::metre * mp_units::si::metre>;

/// The position of a target, the output of the models: three lengths.
using Position = Vector<Length, Length, Length>;
using PositionCovariance = Matrix<Position, Position>;

/// The state of the constant velocity model, position then velocity, and of the constant acceleration model,
/// position, velocity, then acceleration.
using VelocityState = Vector<Length, Length, Length, Speed, Speed, Speed>;
using AccelerationState =
  Vector<Length, Length, Length, Speed, Speed, Speed, Acceleration, Acceleration, Acceleration>;

/// The typed position of a measurement, the boundary with the untyped messages.
[[nodiscard]] inline Position position_of(const sensor_measurement & z)
{
  using mp_units::si::metre;
  return Position{z.z(0) * metre, z.z(1) * metre, z.z(2) * metre};
}

/// The determinant of a 3x3 typed covariance, by cofactors, typed as the product of its diagonal.
template <typename Covariance>
[[nodiscard]] auto determinant3(const Covariance & m)
{
  return m.template at<0, 0>() * (m.template at<1, 1>() * m.template at<2, 2>() -
         m.template at<2, 1>() * m.template at<1, 2>()) -
         m.template at<1, 0>() * (m.template at<0, 1>() * m.template at<2, 2>() -
         m.template at<2, 1>() * m.template at<0, 2>()) +
         m.template at<2, 0>() * (m.template at<0, 1>() * m.template at<1, 2>() -
         m.template at<1, 1>() * m.template at<0, 2>());
}

/// A motion model of the given state, `Derivatives` the count of position derivatives it carries: 2 for the
/// constant velocity model, 3 for the constant acceleration model.
template <typename State, std::size_t Derivatives>
class AxesModel
{
public:
  using Covariance = Matrix<State, State>;
  using OutputModel = kf::evaluate<kf::quotient<Position, State>>;
  using StateTransition = kf::evaluate<kf::quotient<State, State>>;
  static constexpr std::size_t states{3 * Derivatives};

  /// The filter of a track: the state, its covariance, and the original models' simple update form.
  static auto make_filter()
  {
    OutputModel h{};
    [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
      (h.template at<Axis, Axis>(1. * mp_units::one), ...);
    }(std::make_index_sequence<3>{});
    return fcarouge::kalman{fcarouge::state{State{}}, fcarouge::output<Position>,
                            fcarouge::estimate_uncertainty{Covariance{}},
                            fcarouge::process_uncertainty{Covariance{}},
                            fcarouge::output_uncertainty{PositionCovariance{}},
                            fcarouge::output_model{h}, fcarouge::state_transition{StateTransition{}},
                            fcarouge::simple_form};
  }

  using filter_type = decltype(make_filter());

  /// A buffered estimate, to rewind a track to the time of a measurement.
  struct snapshot
  {
    double time_stamp;
    State x;
    Covariance P;
  };

  /// The spectral density of the white noise driving the highest derivative: the original's sigma_a for the
  /// constant velocity model [m/s^1.5], sigma_j for the constant acceleration model [m/s^2.5], as a number.
  void spectral_density(double sigma) { sigma_ = sigma; }

  /// The initial covariance of a track from the first measurements: the original constant velocity model
  /// leaves it zero (its P(sigma_p, sigma_v) is never called); the constant acceleration model's is diagonal.
  void initial_standard_deviations(double sigma_p, double sigma_v, double sigma_a)
  {
    initial_ = Covariance{};
    if constexpr (Derivatives == 3) {
      using mp_units::si::metre;
      using mp_units::si::second;
      [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
        (initial_.template at<Axis, Axis>(sigma_p * sigma_p * Element<Covariance, Axis, Axis>::reference), ...);
        (initial_.template at<Axis + 3, Axis + 3>(
           sigma_v * sigma_v * Element<Covariance, Axis + 3, Axis + 3>::reference), ...);
        (initial_.template at<Axis + 6, Axis + 6>(
           sigma_a * sigma_a * Element<Covariance, Axis + 6, Axis + 6>::reference), ...);
      }(std::make_index_sequence<3>{});
    } else {
      (void)sigma_p;
      (void)sigma_v;
      (void)sigma_a;
    }
  }

  /// The output covariance R, diagonal, from the original's r_diag [m²].
  bool output_variances(const std::vector<double> & r_diag)
  {
    if (r_diag.size() != 3) {
      return false;
    }
    r_ = PositionCovariance{};
    [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
      (r_.template at<Axis, Axis>(r_diag[Axis] * Element<PositionCovariance, Axis, Axis>::reference), ...);
    }(std::make_index_sequence<3>{});
    return true;
  }

  /// The state transition over dt [s], the identity with the integrals of the derivatives.
  [[nodiscard]] static StateTransition transition(double dt)
  {
    StateTransition f{};
    [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
      ((f.template at<Axis, Axis>(1. * mp_units::one), f.template at<Axis + 3, Axis + 3>(1. * mp_units::one),
        f.template at<Axis, Axis + 3>(dt * Element<StateTransition, Axis, Axis + 3>::reference)),
       ...);
      if constexpr (Derivatives == 3) {
        ((f.template at<Axis + 6, Axis + 6>(1. * mp_units::one),
          f.template at<Axis, Axis + 6>(0.5 * dt * dt * Element<StateTransition, Axis, Axis + 6>::reference),
          f.template at<Axis + 3, Axis + 6>(dt * Element<StateTransition, Axis + 3, Axis + 6>::reference)),
         ...);
      }
    }(std::make_index_sequence<3>{});
    return f;
  }

  /// The process noise over dt [s] of a white noise of the configured spectral density: the original's Q(dt),
  /// each element computed in its order of operations, then given its unit.
  [[nodiscard]] Covariance process_noise(double dt) const
  {
    Covariance q{};
    const double s2{sigma_ * sigma_};
    auto set = [&]<std::size_t Row, std::size_t Column>(double value) {
      q.template at<Row, Column>(s2 * value * Element<Covariance, Row, Column>::reference);
    };
    [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
      if constexpr (Derivatives == 2) {
        (set.template operator()<Axis, Axis>(dt * dt * dt / 3), ...);
        (set.template operator()<Axis, Axis + 3>(dt * dt / 2), ...);
        (set.template operator()<Axis + 3, Axis>(dt * dt / 2), ...);
        (set.template operator()<Axis + 3, Axis + 3>(dt), ...);
      } else {
        const double dt3{dt * dt * dt}, dt4{dt3 * dt}, dt5{dt4 * dt};
        (set.template operator()<Axis, Axis>(dt5 / 20.0), ...);
        (set.template operator()<Axis, Axis + 3>(dt4 / 8.0), ...);
        (set.template operator()<Axis + 3, Axis>(dt4 / 8.0), ...);
        (set.template operator()<Axis, Axis + 6>(dt3 / 6.0), ...);
        (set.template operator()<Axis + 6, Axis>(dt3 / 6.0), ...);
        (set.template operator()<Axis + 3, Axis + 3>(dt3 / 3.0), ...);
        (set.template operator()<Axis + 3, Axis + 6>(dt * dt / 2.0), ...);
        (set.template operator()<Axis + 6, Axis + 3>(dt * dt / 2.0), ...);
        (set.template operator()<Axis + 6, Axis + 6>(dt), ...);
      }
    }(std::make_index_sequence<3>{});
    return q;
  }

  /// A track's filter from a measurement: its position, the other derivatives zero, the initial covariance.
  [[nodiscard]] filter_type init(const sensor_measurement & z) const
  {
    filter_type filter{make_filter()};
    State x{};
    const Position p{position_of(z)};
    [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
      (x.template at<Axis>(p.template at<Axis>()), ...);
    }(std::make_index_sequence<3>{});
    filter.x(x);
    filter.p(initial_);
    filter.r(r_);
    return filter;
  }

  /// A track's filter born from an unassigned measurement: the original sets its covariance to Q(dt_pred).
  [[nodiscard]] filter_type birth(const sensor_measurement & z, double dt_pred) const
  {
    filter_type filter{init(z)};
    filter.p(process_noise(dt_pred));
    return filter;
  }

  /// The prediction over dt [s], skipped and reported for a non-positive dt, as the original.
  bool predict(filter_type & filter, double dt)
  {
    constexpr const char * where{Derivatives == 2 ? "ConstVel::predictX" : "ConstAccel::predictX"};
    if (dt <= 0) {
      reportNonPositiveDt(predict_dt_warning_, where, dt);
      return false;
    }
    clearNonPositiveDt(predict_dt_warning_, where);
    filter.f(transition(dt));
    filter.q(process_noise(dt));
    filter.predict();
    return true;
  }

  /// The correction by a position measurement, of the configured output covariance, as the original.
  void update(filter_type & filter, const sensor_measurement & z) const
  {
    filter.r(r_);
    filter.update(position_of(z));
  }

  /// The innovation y = z - H x and its covariance S = R + H P Hᵀ of a candidate measurement, without changing
  /// the filter.
  [[nodiscard]] std::pair<Position, PositionCovariance> innovation(
    const filter_type & filter,
    const sensor_measurement & z) const
  {
    const OutputModel h{filter.h()};
    return {Position{position_of(z) - h * filter.x()},
            PositionCovariance{r_ + h * filter.p() * fcarouge::transposed(h)}};
  }

  /// The log-likelihood of a candidate measurement: -½ (yᵀ S⁻¹ y + ln(det S / 1 m⁶) + 3 ln 2π). The original
  /// takes the logarithm of a determinant in m⁶ implicitly; the unit is explicit here.
  [[nodiscard]] double log_likelihood(const filter_type & filter, const sensor_measurement & z) const
  {
    const auto [y, s] = innovation(filter, z);
    const auto det_s{determinant3(s)};
    if (det_s <= 0 * det_s.reference) {
      std::printf("ERROR [logLikelihood] Non-positive definite S matrix. det(S) = %f\n",
                  det_s.numerical_value_in(det_s.unit));
      return -std::numeric_limits<double>::infinity();
    }
    const auto mahalanobis{fcarouge::transposed(y) / s * y};
    // yᵀ S⁻¹ y is a 1x1 typed matrix, a single quantity: rank 0, read without position
    const double squared{mahalanobis.at().numerical_value_in(mp_units::one)};
    return -0.5 * (squared +
                   std::log(det_s.numerical_value_in(mp_units::si::metre * mp_units::si::metre * mp_units::si::metre * mp_units::si::metre * mp_units::si::metre * mp_units::si::metre)) +
                   3 * std::log(2 * std::numbers::pi));
  }

  /// The volume of the position uncertainty, the square root of the determinant of its covariance [m³].
  [[nodiscard]] static Volume position_volume(const filter_type & filter)
  {
    const Covariance p{filter.p()};
    PositionCovariance block{};
    [&]<std::size_t... Index>(std::index_sequence<Index...>) {
      (block.template at<Index / 3, Index % 3>(p.template at<Index / 3, Index % 3>()), ...);
    }(std::make_index_sequence<9>{});
    return Volume{mp_units::sqrt(mp_units::abs(determinant3(block)))};
  }

  /// The Euclidean norm of the state's numerical values in SI units, as the original's x.norm(), which adds
  /// lengths, speeds, and accelerations: meaningful only as a test for infinite values.
  [[nodiscard]] static double state_norm(const filter_type & filter)
  {
    double norm2{0};
    for (const double v : state_values(filter)) {
      norm2 += v * v;
    }
    return std::sqrt(norm2);
  }

  /// Whether the state norm is finite, as the original's isinf(x.norm()) test.
  [[nodiscard]] static bool finite(const filter_type & filter) { return !std::isinf(state_norm(filter)); }

  [[nodiscard]] static snapshot save(const filter_type & filter, double time_stamp)
  {
    return {time_stamp, State{filter.x()}, Covariance{filter.p()}};
  }

  static void restore(filter_type & filter, const snapshot & s)
  {
    filter.x(s.x);
    filter.p(s.P);
  }

  /// The state, in SI units, for the messages: m, m/s, m/s².
  [[nodiscard]] static std::array<double, states> state_values(const filter_type & filter)
  {
    const State x{filter.x()};
    std::array<double, states> values{};
    [&]<std::size_t... Index>(std::index_sequence<Index...>) {
      ((values[Index] = value_of<Index>(x)), ...);
    }(std::make_index_sequence<states>{});
    return values;
  }

  /// The covariance, row major, in SI units, for the messages.
  [[nodiscard]] static std::array<double, states * states> covariance_values(const filter_type & filter)
  {
    const Covariance p{filter.p()};
    std::array<double, states * states> values{};
    [&]<std::size_t... Index>(std::index_sequence<Index...>) {
      ((values[Index] = value_of<Index / states, Index % states>(p)), ...);
    }(std::make_index_sequence<states * states>{});
    return values;
  }

private:
  double sigma_{0};
  DtWarnState predict_dt_warning_{};
  Covariance initial_{};
  PositionCovariance r_{};
};

using ConstantVelocity = AxesModel<VelocityState, 2>;
using ConstantAcceleration = AxesModel<AccelerationState, 3>;
}  // namespace multi_target_kf::typed

#endif  // TYPED_MODELS_H
