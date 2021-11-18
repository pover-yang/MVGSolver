#ifndef MVGSOLVER_SOLVER_BASE_H
#define MVGSOLVER_SOLVER_BASE_H

#include <cstddef>
#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>

#include "model.h"

namespace solver {
    // This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
    class SolverBase {
    public:
        SolverBase() = default;

        ~SolverBase() = default;

        // The minimum number of points required for the estimation
        virtual size_t sampleSize() {
            return 0;
        }

        // The maximum number of solutions returned by the estimator
        virtual size_t maximumSolutions() {
            return 1;
        }

        // Determines if there is a chance of returning multiple models
        virtual bool returnMultipleModels() {
            return maximumSolutions() > 1;
        }
    };
}

#endif //MVGSOLVER_SOLVER_BASE_H
