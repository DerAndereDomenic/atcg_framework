#pragma once

#include <Registration/Registration.h>
#include <Eigen/SVD>
#include <Registration/CPDBackend.h>

namespace atcg
{
    class CoherentPointDrift : public Registration
    {
    public:
        CoherentPointDrift(const std::shared_ptr<PointCloud>& source, const std::shared_ptr<PointCloud>& target, const double& w = 0.0);

        virtual ~CoherentPointDrift();

        virtual void solve(const uint32_t& maxN, const float& tol = 0.01f) override;

        virtual void applyTransform(const std::shared_ptr<PointCloud>& cloud) override;
    private:
        double initialize();
        void estimate(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double var);
        double maximize(Eigen::VectorXd& PX, Eigen::VectorXd& PY);

        Transformation T;
        
        double w = 0.0;
        double Np = 0.0;

        Eigen::JacobiSVD<RowMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;

        std::unique_ptr<CPDBackend> _backend;
    };
}