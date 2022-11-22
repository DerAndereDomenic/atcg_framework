#pragma once

#include <Eigen/Dense>

namespace atcg
{
    struct Transformation
    {
        double s = 1.0;
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
    };
}