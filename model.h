#pragma once

#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>


class Model {
public:
    Eigen::MatrixXd descriptor; // The descriptor of the current model
    explicit Model(Eigen::MatrixXd descriptor_) : descriptor(std::move(descriptor_)) {}

    Model() = default;
};


class FundamentalMatrix : public Model {
public:
    FundamentalMatrix() : Model(Eigen::Matrix3d()) {}

    FundamentalMatrix(const FundamentalMatrix &other) : Model(other) {
        descriptor = other.descriptor;
    }
};

class EssentialMatrix : public Model {
public:
    EssentialMatrix() : Model(Eigen::Matrix3d()) {}

    EssentialMatrix(const EssentialMatrix &other) : Model(other) {
        descriptor = other.descriptor;
    }
};


class Homography : public Model {
public:
    Homography() : Model(Eigen::Matrix3d()) {}

    Homography(const Homography &other) : Model(other) {
        descriptor = other.descriptor;
    }
};

