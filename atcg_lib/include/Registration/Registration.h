#pragma once

#include <DataStructure/PointCloud.h>

namespace atcg
{
class Registration
{
public:
    Registration(const std::shared_ptr<PointCloud>& source, const std::shared_ptr<PointCloud>& target);

    virtual ~Registration() {};

    virtual void solve(const uint32_t& maxN, const float& tol = 0.01f) = 0;

    virtual void applyTransform(const std::shared_ptr<PointCloud>& cloud) = 0;

protected:
    RowMatrix X, Y;
    uint32_t N, M;
};
}    // namespace atcg