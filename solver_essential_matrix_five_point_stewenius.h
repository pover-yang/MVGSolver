#include "solver_base.h"

namespace solver {
    class EssentialMatrixFivePointSolverStewenius : SolverBase {
    public:
        EssentialMatrixFivePointSolverStewenius() = default;

        ~EssentialMatrixFivePointSolverStewenius() = default;

        constexpr size_t sampleSize() override;

        constexpr size_t maximumSolutions() override;

        /*!
         * Estimate the model parameters from the given point sample using weighted fitting if possible.
         * @param data The set of data points
         * @param sample The sample used for the estimation
         * @param models The estimated model parameters
         * @param weights The weight for each point
         * @return
         */
        bool estimateModel(const cv::Mat &data, const size_t *sample, std::vector<Model> &models,
                           const double *weights = nullptr);

    protected:
        static Eigen::Matrix<double, 1, 10> multiplyDegOnePoly(
                const Eigen::RowVector4d &a,
                const Eigen::RowVector4d &b);

        static Eigen::Matrix<double, 1, 20> multiplyDegTwoDegOnePoly(
                const Eigen::Matrix<double, 1, 10> &a,
                const Eigen::RowVector4d &b);

        static Eigen::Matrix<double, 10, 20> buildConstraintMatrix(
                const Eigen::Matrix<double, 1, 4> nullSpace[3][3]);

        static Eigen::Matrix<double, 9, 20> getTraceConstraint(
                const Eigen::Matrix<double, 1, 4> nullSpace[3][3]);

        static Eigen::Matrix<double, 1, 10>
        computeEETranspose(const Eigen::Matrix<double, 1, 4> nullSpace[3][3], int i, int j);

        static Eigen::Matrix<double, 1, 20> getDeterminantConstraint(
                const Eigen::Matrix<double, 1, 4> nullSpace[3][3]);
    };
}