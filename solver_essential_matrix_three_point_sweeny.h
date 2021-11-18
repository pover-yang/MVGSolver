#include "solver_base.h"

namespace solver {
    class EssentialMatrixThreePointSolverSweeny : SolverBase {
    public:
        EssentialMatrixThreePointSolverSweeny() = default;

        ~EssentialMatrixThreePointSolverSweeny() = default;

        constexpr size_t sampleSize() override;

        constexpr size_t maximumSolutions() override;

        /*!
         *
         * @param data The set of data points
         * @param sample The sample used for the estimation
         * @param models The estimated model parameters
         * @return
         */
        static bool estimateModel(const cv::Mat &data, const size_t *sample, std::vector<Model> &models);

    protected:
        static void essential_from_motion(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, Eigen::Matrix3d *E);

        static int qep_div_1_q2(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B, const Eigen::Matrix3d &C,
                                double eig_vals[4], Eigen::Matrix<double, 3, 4> *eig_vecs);

        static int solve_quartic_real(double b, double c, double d, double e, double roots[4]);

        static void solve_cubic_single_real(double c2, double c1, double c0, double &root);

        static void detpoly3(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B, double coeffs[7]);

        static double sign(double z);
    };
}
