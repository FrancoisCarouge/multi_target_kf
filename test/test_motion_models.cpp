// test/test_motion_models.cpp
#include <gtest/gtest.h>
#include "multi_target_kf/typed_models.h"

#include <array>
#include <cstddef>
#include <vector>

using multi_target_kf::typed::ConstantAcceleration;
using multi_target_kf::typed::ConstantVelocity;

namespace {

// A filter of the model at the given state values, in SI units, with an identity covariance in SI units.
template <typename Model, std::size_t States>
typename Model::filter_type makeFilter(const std::array<double, States>& values) {
    static_assert(States == Model::states);
    using Covariance = typename Model::Covariance;
    using State = std::remove_cvref_t<decltype(std::declval<typename Model::filter_type>().x())>;
    auto filter = Model::make_filter();
    State x{};
    Covariance p{};
    [&]<std::size_t... Index>(std::index_sequence<Index...>) {
        (x.template at<Index>(values[Index] * multi_target_kf::typed::Element<State, Index>::reference), ...);
        (p.template at<Index, Index>(1.0 * multi_target_kf::typed::Element<Covariance, Index, Index>::reference), ...);
    }(std::make_index_sequence<States>{});
    filter.x(x);
    filter.p(p);
    return filter;
}

// A position measurement.
sensor_measurement makeMeasurement(double time_stamp, double x, double y, double z) {
    sensor_measurement meas;
    meas.time_stamp = time_stamp;
    meas.id = 0;
    meas.z = Eigen::VectorXd::Zero(3);
    meas.z << x, y, z;
    meas.R = 0.01 * Eigen::MatrixXd::Identity(3, 3);
    return meas;
}

}  // namespace

TEST(MotionModelTest, ConstantVelocityModelCreation) {
    EXPECT_EQ(ConstantVelocity::states, 6u);
    EXPECT_EQ(multi_target_kf::typed::Position::rows, 3u);
}

TEST(MotionModelTest, ConstantAccelerationModelCreation) {
    EXPECT_EQ(ConstantAcceleration::states, 9u);
    EXPECT_EQ(multi_target_kf::typed::Position::rows, 3u);
}

TEST(MotionModelTest, ConstantVelocityPrediction) {
    ConstantVelocity model;
    model.spectral_density(1e-9);  // a negligible process noise: the transition alone

    // Initialize a state vector with known position and velocity
    auto filter = makeFilter<ConstantVelocity>(std::array{1.0, 2.0, 3.0, 0.5, 0.6, 0.7}); // [x, y, z, vx, vy, vz]

    // Predict one second ahead
    ASSERT_TRUE(model.predict(filter, 1.0));
    const auto x_pred = ConstantVelocity::state_values(filter);

    // Expected result: position += velocity * dt
    EXPECT_NEAR(x_pred[0], 1.5, 1e-5); // x + vx*dt
    EXPECT_NEAR(x_pred[1], 2.6, 1e-5); // y + vy*dt
    EXPECT_NEAR(x_pred[2], 3.7, 1e-5); // z + vz*dt

    // Velocity should remain unchanged
    EXPECT_NEAR(x_pred[3], 0.5, 1e-5);
    EXPECT_NEAR(x_pred[4], 0.6, 1e-5);
    EXPECT_NEAR(x_pred[5], 0.7, 1e-5);
}

TEST(MotionModelTest, ConstantAccelerationPrediction) {
    ConstantAcceleration model;
    model.spectral_density(1e-9);

    // Initialize a state vector with known position, velocity, and acceleration
    auto filter = makeFilter<ConstantAcceleration>(
        std::array{1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3}); // [x, y, z, vx, vy, vz, ax, ay, az]

    // Predict one second ahead
    ASSERT_TRUE(model.predict(filter, 1.0));
    const auto x_pred = ConstantAcceleration::state_values(filter);

    // Expected result: position += velocity * dt + 0.5 * acceleration * dt^2
    EXPECT_NEAR(x_pred[0], 1.5 + 0.5 * 0.1, 1e-5); // x + vx*dt + 0.5*ax*dt^2
    EXPECT_NEAR(x_pred[1], 2.6 + 0.5 * 0.2, 1e-5); // y + vy*dt + 0.5*ay*dt^2
    EXPECT_NEAR(x_pred[2], 3.7 + 0.5 * 0.3, 1e-5); // z + vz*dt + 0.5*az*dt^2

    // Velocity += acceleration * dt
    EXPECT_NEAR(x_pred[3], 0.5 + 0.1, 1e-5); // vx + ax*dt
    EXPECT_NEAR(x_pred[4], 0.6 + 0.2, 1e-5); // vy + ay*dt
    EXPECT_NEAR(x_pred[5], 0.7 + 0.3, 1e-5); // vz + az*dt

    // Acceleration should remain unchanged
    EXPECT_NEAR(x_pred[6], 0.1, 1e-5);
    EXPECT_NEAR(x_pred[7], 0.2, 1e-5);
    EXPECT_NEAR(x_pred[8], 0.3, 1e-5);
}

TEST(MotionModelTest, ConstantVelocityMeasurement) {
    // Initialize a state vector
    auto filter = makeFilter<ConstantVelocity>(std::array{1.0, 2.0, 3.0, 0.5, 0.6, 0.7}); // [x, y, z, vx, vy, vz]

    // Compute expected measurement: only position is observed
    const multi_target_kf::typed::Position z{filter.h() * filter.x()};
    EXPECT_NEAR(multi_target_kf::typed::value_of<0>(z), 1.0, 1e-5); // x
    EXPECT_NEAR(multi_target_kf::typed::value_of<1>(z), 2.0, 1e-5); // y
    EXPECT_NEAR(multi_target_kf::typed::value_of<2>(z), 3.0, 1e-5); // z
}

TEST(MotionModelTest, ConstantAccelerationMeasurement) {
    // Initialize a state vector
    auto filter = makeFilter<ConstantAcceleration>(
        std::array{1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3}); // [x, y, z, vx, vy, vz, ax, ay, az]

    // Compute expected measurement: only position is observed
    const multi_target_kf::typed::Position z{filter.h() * filter.x()};
    EXPECT_NEAR(multi_target_kf::typed::value_of<0>(z), 1.0, 1e-5); // x
    EXPECT_NEAR(multi_target_kf::typed::value_of<1>(z), 2.0, 1e-5); // y
    EXPECT_NEAR(multi_target_kf::typed::value_of<2>(z), 3.0, 1e-5); // z
}

TEST(MotionModelTest, ConstantVelocityPredictAndUpdate) {
    // Set up model parameters
    ConstantVelocity model;
    model.spectral_density(0.1);
    model.output_variances(std::vector<double>{0.01, 0.01, 0.01});

    // Initialize state
    auto filter = makeFilter<ConstantVelocity>(std::array{1.0, 2.0, 3.0, 0.5, 0.6, 0.7}); // [x, y, z, vx, vy, vz]

    // Predict
    ASSERT_TRUE(model.predict(filter, 1.0));

    // Create a measurement with some noise, slightly different from the prediction
    const sensor_measurement meas = makeMeasurement(1.0, 1.6, 2.7, 3.8);

    // Update
    model.update(filter, meas);
    const auto updated = ConstantVelocity::state_values(filter);

    // The updated state should be closer to the measurement than the prediction
    EXPECT_NEAR(updated[0], meas.z(0), 0.1);
    EXPECT_NEAR(updated[1], meas.z(1), 0.1);
    EXPECT_NEAR(updated[2], meas.z(2), 0.1);
}

TEST(MotionModelTest, ConstantAccelerationPredictAndUpdate) {
    // Set up model parameters
    ConstantAcceleration model;
    model.spectral_density(0.1);
    model.output_variances(std::vector<double>{0.01, 0.01, 0.01});

    // Initialize state
    auto filter = makeFilter<ConstantAcceleration>(
        std::array{1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3}); // [x, y, z, vx, vy, vz, ax, ay, az]

    // Predict
    ASSERT_TRUE(model.predict(filter, 1.0));

    // Create a measurement with some noise, slightly different from the prediction
    const sensor_measurement meas = makeMeasurement(1.0, 1.65, 2.8, 3.95);

    // Update
    model.update(filter, meas);
    const auto updated = ConstantAcceleration::state_values(filter);

    // The updated state should be closer to the measurement than the prediction
    EXPECT_NEAR(updated[0], meas.z(0), 0.1);
    EXPECT_NEAR(updated[1], meas.z(1), 0.1);
    EXPECT_NEAR(updated[2], meas.z(2), 0.1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
