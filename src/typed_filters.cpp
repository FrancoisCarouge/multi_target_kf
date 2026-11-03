// The constant velocity and constant acceleration filters on the FrancoisCarouge Kalman library, with
// physical units on every element, in the original filters' simple form (I - K H) P.
#include "multi_target_kf/typed_filters.h"

#include <fcarouge/eigen.hpp>
#include <fcarouge/kalman.hpp>
#include <fcarouge/mp_units.hpp>
#include <eigen/fcarouge/kalman_eigen.hpp>
#include <mp_units/fcarouge/kalman_mp_units.hpp>
#include <typed/fcarouge/kalman_typed_linear_algebra.hpp>
#include <mp-units/systems/si.h>

#include <cstddef>
#include <utility>

namespace multi_target_kf {
namespace {

namespace kf = fcarouge::kalman_filter::internal;

constexpr auto m = mp_units::si::metre;
constexpr auto s = mp_units::si::second;
constexpr auto one = mp_units::one;

template <auto Reference> using Quantity = mp_units::quantity<Reference, double>;
using Position = Quantity<m>;
using Velocity = Quantity<m / s>;
using Acceleration = Quantity<m / (s * s)>;
using Time = Quantity<s>;

template <typename... Types>
using ColumnVector = fcarouge::typed_column_vector<Eigen::Matrix<double, sizeof...(Types), 1>, Types...>;

// the types of a filter of the state, observing its first three positions
template <typename State>
struct Types {
    using Output = ColumnVector<Position, Position, Position>;
    using Covariance = kf::evaluate<kf::product<State, kf::evaluate<kf::transpose<State>>>>;
    using OutputVariance = kf::evaluate<kf::product<Output, kf::evaluate<kf::transpose<Output>>>>;
    using OutputModel = kf::evaluate<kf::quotient<Output, State>>;
    using StateTransition = kf::evaluate<kf::quotient<State, State>>;
};

// calls callable.template operator()<axis>() for the three axes
template <typename Callable>
constexpr void forAxes(Callable &&callable)
{
    callable.template operator()<0>();
    callable.template operator()<1>();
    callable.template operator()<2>();
}

template <typename State>
auto makeFilter()
{
    using namespace fcarouge;
    using T = Types<State>;
    // H, the three positions of the state
    typename T::OutputModel h{};
    forAxes([&]<std::size_t axis> { h.template at<axis, axis>(1. * one); });
    return kalman{state{State{}},
                  output<typename T::Output>,
                  estimate_uncertainty{typename T::Covariance{}},
                  process_uncertainty{typename T::Covariance{}},
                  output_uncertainty{typename T::OutputVariance{}},
                  output_model{h},
                  state_transition{typename T::StateTransition{}},
                  // the covariance corrected as (I - K H) P, as the original
                  simple_form};
}

// the filter of a state, its characteristics set from the track's before each step: one model serves all tracks
template <typename State>
auto &filter()
{
    thread_local auto instance = makeFilter<State>();
    return instance;
}

template <typename State>
void store(const auto &kalman, Eigen::VectorXd &x, Eigen::MatrixXd &P)
{
    x = kalman.x().data();
    P = kalman.p().data();
}

template <typename State, typename Dimensions>
void update(Eigen::VectorXd &x, Eigen::MatrixXd &P, const Eigen::VectorXd &z, const Eigen::MatrixXd &R)
{
    using T = Types<State>;
    auto &kalman = filter<State>();
    kalman.x(State{Eigen::Matrix<double, Dimensions::value, 1>{x}});
    kalman.p(typename T::Covariance{Eigen::Matrix<double, Dimensions::value, Dimensions::value>{P}});
    kalman.r(typename T::OutputVariance{Eigen::Matrix3d{R}});
    kalman.update(typename T::Output{Eigen::Vector3d{z}});
    store<State>(kalman, x, P);
}

using ConstantVelocity = ColumnVector<Position, Position, Position, Velocity, Velocity, Velocity>;
using ConstantAcceleration = ColumnVector<Position, Position, Position, Velocity, Velocity, Velocity,
                                          Acceleration, Acceleration, Acceleration>;

} // namespace

void predictConstantVelocity(Eigen::VectorXd &x, Eigen::MatrixXd &P, double dt, double spectral_acceleration)
{
    using State = ConstantVelocity;
    using T = Types<State>;
    const Time t = dt * s;
    // the white acceleration's spectral density
    const auto density = spectral_acceleration * spectral_acceleration * (m * m / (s * s * s));

    typename T::StateTransition f{};
    typename T::Covariance q{};
    forAxes([&]<std::size_t axis> {
        f.template at<axis, axis>(1. * one);
        f.template at<axis + 3, axis + 3>(1. * one);
        f.template at<axis, axis + 3>(t);
        q.template at<axis, axis>(density * t * t * t / 3.);
        q.template at<axis, axis + 3>(density * t * t / 2.);
        q.template at<axis + 3, axis>(density * t * t / 2.);
        q.template at<axis + 3, axis + 3>(density * t);
    });

    auto &kalman = filter<State>();
    kalman.x(State{Eigen::Matrix<double, 6, 1>{x}});
    kalman.p(typename T::Covariance{Eigen::Matrix<double, 6, 6>{P}});
    kalman.f(f);
    kalman.q(q);
    kalman.predict();
    store<State>(kalman, x, P);
}

void predictConstantAcceleration(Eigen::VectorXd &x, Eigen::MatrixXd &P, double dt, double spectral_jerk)
{
    using State = ConstantAcceleration;
    using T = Types<State>;
    const Time t = dt * s;
    // the white jerk's spectral density
    const auto density = spectral_jerk * spectral_jerk * (m * m / (s * s * s * s * s));

    typename T::StateTransition f{};
    typename T::Covariance q{};
    forAxes([&]<std::size_t axis> {
        f.template at<axis, axis>(1. * one);
        f.template at<axis + 3, axis + 3>(1. * one);
        f.template at<axis + 6, axis + 6>(1. * one);
        f.template at<axis, axis + 3>(t);
        f.template at<axis, axis + 6>(t * t / 2.);
        f.template at<axis + 3, axis + 6>(t);
        q.template at<axis, axis>(density * t * t * t * t * t / 20.);
        q.template at<axis, axis + 3>(density * t * t * t * t / 8.);
        q.template at<axis + 3, axis>(density * t * t * t * t / 8.);
        q.template at<axis, axis + 6>(density * t * t * t / 6.);
        q.template at<axis + 6, axis>(density * t * t * t / 6.);
        q.template at<axis + 3, axis + 3>(density * t * t * t / 3.);
        q.template at<axis + 3, axis + 6>(density * t * t / 2.);
        q.template at<axis + 6, axis + 3>(density * t * t / 2.);
        q.template at<axis + 6, axis + 6>(density * t);
    });

    auto &kalman = filter<State>();
    kalman.x(State{Eigen::Matrix<double, 9, 1>{x}});
    kalman.p(typename T::Covariance{Eigen::Matrix<double, 9, 9>{P}});
    kalman.f(f);
    kalman.q(q);
    kalman.predict();
    store<State>(kalman, x, P);
}

void updateConstantVelocity(Eigen::VectorXd &x, Eigen::MatrixXd &P, const Eigen::VectorXd &z, const Eigen::MatrixXd &R)
{
    update<ConstantVelocity, std::integral_constant<int, 6>>(x, P, z, R);
}

void updateConstantAcceleration(Eigen::VectorXd &x, Eigen::MatrixXd &P, const Eigen::VectorXd &z, const Eigen::MatrixXd &R)
{
    update<ConstantAcceleration, std::integral_constant<int, 9>>(x, P, z, R);
}

} // namespace multi_target_kf
