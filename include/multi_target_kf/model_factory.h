// include/multi_target_kf/model_factory.h
#ifndef MODEL_FACTORY_H
#define MODEL_FACTORY_H

/**
 * @brief Enumeration of available motion model types
 */
enum ModelType {
    CONSTANT_VELOCITY = 0,
    CONSTANT_ACCELERATION = 1,
    ADAPTIVE_ACCEL_UKF = 2,
    // Add more model types here as they become available
};

/**
 * @brief The names of the motion model types; a tracker of a model type is made by makeTracker() in
 * kf_tracker.h, each model its own tracker type
 */
class ModelFactory {
public:
    /**
     * @brief Get the name of a model type
     * @param type The model type
     * @return String representation of the model type
     */
    static const char* getModelName(ModelType type) {
        switch (type) {
            case CONSTANT_VELOCITY:
                return "Constant Velocity";
            case CONSTANT_ACCELERATION:
                return "Constant Acceleration";
            case ADAPTIVE_ACCEL_UKF:
                return "Adaptive Acceleration UKF";
            // Add more cases for future models
            default:
                return "Unknown Model";
        }
    }
};

#endif // MODEL_FACTORY_H