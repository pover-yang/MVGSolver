#include "solver_essential_matrix_five_point_stewenius.h"

namespace solver {
    constexpr size_t EssentialMatrixFivePointSolverStewenius::sampleSize() {
        return 5;
    }

    constexpr size_t EssentialMatrixFivePointSolverStewenius::maximumSolutions() {
        return 10;
    }

    bool EssentialMatrixFivePointSolverStewenius::estimateModel(const cv::Mat &data, const size_t *sample,
                                                                std::vector<Model> &models, const double *weights_) {
        size_t sample_size = sampleSize();
        if (sample == nullptr) { sample_size = data.rows; }

        Eigen::MatrixXd coefficients(sample_size, 9);
        const double *data_ptr = reinterpret_cast<double *>(data.data);
        const size_t cols = data.cols;

        // Step 1. Create the nx9 matrix containing epipolar constraints.
        //   Essential matrix is a linear combination of the 4 vectors spanning the null space of this matrix.
        size_t offset;
        double x0, y0, x1, y1, weight = 1.0;
        for (size_t i = 0; i < sample_size; i++) {
            if (sample == nullptr) {
                offset = cols * i;
                if (weights_ != nullptr) {
                    weight = weights_[i];
                }
            } else {
                offset = cols * sample[i];
                if (weights_ != nullptr) {
                    weight = weights_[sample[i]];
                }
            }

            // Precalculate these values to avoid calculating them multiple times
            const double weight_times_x0 = data_ptr[offset] * weight;
            const double weight_times_y0 = data_ptr[offset + 1] * weight;
            const double weight_times_x1 = data_ptr[offset + 2] * weight;
            const double weight_times_y1 = data_ptr[offset + 3] * weight;

            coefficients.row(int(i)) << weight_times_x0 * x1, weight_times_x0 * y1, weight_times_x0,
                    weight_times_y0 * x1, weight_times_y0 * y1, weight_times_y0,
                    weight_times_x1, weight_times_y1, weight;
        }

        // Extract the null space from a minimal sampling (using LU) or non-minimal sampling (using SVD).
        Eigen::Matrix<double, 9, 4> nullSpace;

        if (sample_size == 5) {
            const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
            if (lu.dimensionOfKernel() != 4) {
                return false;
            }
            nullSpace = lu.kernel();
        } else {
            const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(
                    coefficients.transpose() * coefficients);
            const Eigen::MatrixXd &Q = qr.matrixQ();
            nullSpace = Q.rightCols<4>();
        }

        const Eigen::Matrix<double, 1, 4> nullSpaceMatrix[3][3] = {
                {nullSpace.row(0), nullSpace.row(3), nullSpace.row(6)},
                {nullSpace.row(1), nullSpace.row(4), nullSpace.row(7)},
                {nullSpace.row(2), nullSpace.row(5), nullSpace.row(8)}};

        // Step 2. Expansion of the epipolar constraints on the determinant and trace.
        const Eigen::Matrix<double, 10, 20> constraintMatrix = buildConstraintMatrix(nullSpaceMatrix);

        // Step 3. Eliminate part of the matrix to isolate polynomials in z.
        Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> c_lu(constraintMatrix.block<10, 10>(0, 0));
        const Eigen::Matrix<double, 10, 10> eliminatedMatrix = c_lu.solve(constraintMatrix.block<10, 10>(0, 10));

        Eigen::Matrix<double, 10, 10> actionMatrix = Eigen::Matrix<double, 10, 10>::Zero();
        actionMatrix.block<3, 10>(0, 0) = eliminatedMatrix.block<3, 10>(0, 0);
        actionMatrix.row(3) = eliminatedMatrix.row(4);
        actionMatrix.row(4) = eliminatedMatrix.row(5);
        actionMatrix.row(5) = eliminatedMatrix.row(7);
        actionMatrix(6, 0) = -1.0;
        actionMatrix(7, 1) = -1.0;
        actionMatrix(8, 3) = -1.0;
        actionMatrix(9, 6) = -1.0;

        Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigensolver(actionMatrix);
        const Eigen::VectorXcd &eigenvalues = eigensolver.eigenvalues();

        // Now that we have x, y, and z we need to substitute them back into the null space to get a valid
        // essential matrix solution.
        for (int i = 0; i < 10; i++) {
            // Only consider real solutions.
            if (eigenvalues(i).imag() != 0) {
                continue;
            }
            Eigen::Matrix3d E_dst_src;
            Eigen::Map<Eigen::Matrix<double, 9, 1 >>(E_dst_src.data()) =
                    nullSpace * eigensolver.eigenvectors().col(i).tail<4>().real();

            EssentialMatrix model;
            model.descriptor = E_dst_src;
            models.push_back(model);
        }

        return !models.empty();
    }

    // Multiply two degree one polynomials of variables x, y, z.
    // E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
    // Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
    Eigen::Matrix<double, 1, 10> EssentialMatrixFivePointSolverStewenius::multiplyDegOnePoly(
            const Eigen::RowVector4d &a,
            const Eigen::RowVector4d &b) {
        Eigen::Matrix<double, 1, 10> output;
        // x^2
        output(0) = a(0) * b(0);
        // xy
        output(1) = a(0) * b(1) + a(1) * b(0);
        // y^2
        output(2) = a(1) * b(1);
        // xz
        output(3) = a(0) * b(2) + a(2) * b(0);
        // yz
        output(4) = a(1) * b(2) + a(2) * b(1);
        // z^2
        output(5) = a(2) * b(2);
        // x
        output(6) = a(0) * b(3) + a(3) * b(0);
        // y
        output(7) = a(1) * b(3) + a(3) * b(1);
        // z
        output(8) = a(2) * b(3) + a(3) * b(2);
        // 1
        output(9) = a(3) * b(3);
        return output;
    }

    // Multiply a 2 deg poly (in x, y, z) and a one deg poly in GrevLex order.
    // x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
    Eigen::Matrix<double, 1, 20> EssentialMatrixFivePointSolverStewenius::multiplyDegTwoDegOnePoly(
            const Eigen::Matrix<double, 1, 10> &a,
            const Eigen::RowVector4d &b) {
        Eigen::Matrix<double, 1, 20> output;
        // x^3
        output(0) = a(0) * b(0);
        // x^2y
        output(1) = a(0) * b(1) + a(1) * b(0);
        // xy^2
        output(2) = a(1) * b(1) + a(2) * b(0);
        // y^3
        output(3) = a(2) * b(1);
        // x^2z
        output(4) = a(0) * b(2) + a(3) * b(0);
        // xyz
        output(5) = a(1) * b(2) + a(3) * b(1) + a(4) * b(0);
        // y^2z
        output(6) = a(2) * b(2) + a(4) * b(1);
        // xz^2
        output(7) = a(3) * b(2) + a(5) * b(0);
        // yz^2
        output(8) = a(4) * b(2) + a(5) * b(1);
        // z^3
        output(9) = a(5) * b(2);
        // x^2
        output(10) = a(0) * b(3) + a(6) * b(0);
        // xy
        output(11) = a(1) * b(3) + a(6) * b(1) + a(7) * b(0);
        // y^2
        output(12) = a(2) * b(3) + a(7) * b(1);
        // xz
        output(13) = a(3) * b(3) + a(6) * b(2) + a(8) * b(0);
        // yz
        output(14) = a(4) * b(3) + a(7) * b(2) + a(8) * b(1);
        // z^2
        output(15) = a(5) * b(3) + a(8) * b(2);
        // x
        output(16) = a(6) * b(3) + a(9) * b(0);
        // y
        output(17) = a(7) * b(3) + a(9) * b(1);
        // z
        output(18) = a(8) * b(3) + a(9) * b(2);
        // 1
        output(19) = a(9) * b(3);
        return output;
    }

    Eigen::Matrix<double, 1, 20> EssentialMatrixFivePointSolverStewenius::getDeterminantConstraint(
            const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) {
        // Singularity constraint.
        return multiplyDegTwoDegOnePoly(
                multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][2]) -
                multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][1]),
                nullSpace[2][0]) +
               multiplyDegTwoDegOnePoly(
                       multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][0]) -
                       multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][2]),
                       nullSpace[2][1]) +
               multiplyDegTwoDegOnePoly(
                       multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][1]) -
                       multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][0]),
                       nullSpace[2][2]);
    }

    // Shorthand for multiplying the Essential matrix with its transpose.
    Eigen::Matrix<double, 1, 10> EssentialMatrixFivePointSolverStewenius::computeEETranspose(
            const Eigen::Matrix<double, 1, 4> nullSpace[3][3], int i, int j) {
        return multiplyDegOnePoly(nullSpace[i][0], nullSpace[j][0]) +
               multiplyDegOnePoly(nullSpace[i][1], nullSpace[j][1]) +
               multiplyDegOnePoly(nullSpace[i][2], nullSpace[j][2]);
    }

    // Builds the trace constraint: EEtE - 1/2 trace(EEt)E = 0
    Eigen::Matrix<double, 9, 20> EssentialMatrixFivePointSolverStewenius::getTraceConstraint(
            const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) {
        Eigen::Matrix<double, 9, 20> traceConstraint;

        // Compute EEt.
        Eigen::Matrix<double, 1, 10> eet[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                eet[i][j] = 2 * computeEETranspose(nullSpace, i, j);
            }
        }

        // Compute the trace.
        const Eigen::Matrix<double, 1, 10> trace = eet[0][0] + eet[1][1] + eet[2][2];

        // Multiply EEt with E.
        for (auto i = 0; i < 3; i++) {
            for (auto j = 0; j < 3; j++) {
                traceConstraint.row(3 * i + j) = multiplyDegTwoDegOnePoly(eet[i][0], nullSpace[0][j]) +
                                                 multiplyDegTwoDegOnePoly(eet[i][1], nullSpace[1][j]) +
                                                 multiplyDegTwoDegOnePoly(eet[i][2], nullSpace[2][j]) -
                                                 0.5 * multiplyDegTwoDegOnePoly(trace, nullSpace[i][j]);
            }
        }

        return traceConstraint;
    }

    Eigen::Matrix<double, 10, 20> EssentialMatrixFivePointSolverStewenius::buildConstraintMatrix(
            const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) {
        Eigen::Matrix<double, 10, 20> constraintMatrix;
        constraintMatrix.block<9, 20>(0, 0) = getTraceConstraint(nullSpace);
        constraintMatrix.row(9) = getDeterminantConstraint(nullSpace);
        return constraintMatrix;
    }
}