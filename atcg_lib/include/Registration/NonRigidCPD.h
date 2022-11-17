#pragma once

#include <Registration/Registration.h>
#include <Eigen/SVD>

namespace atcg
{
    class NonRigidCoherentPointDrift : public Registration
    {
    public:
        NonRigidCoherentPointDrift(const std::shared_ptr<PointCloud>& source, const std::shared_ptr<PointCloud>& target, const double& w = 0.0);

        virtual ~NonRigidCoherentPointDrift();

        virtual void solve(const uint32_t& maxN, const float& tol = 0.01f) override;

        virtual void applyTransform(const std::shared_ptr<PointCloud>& cloud) override;
    private:
        double initialize();
        void estimate(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double var);
        double maximize(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double var);

        void direct_optimization(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double bias, double var);
        double Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var);
        double Pmn(const double& L2S, double var);
        
        double w = 0.0;
        double beta = 0.1;
        double lambda = 0.1;

        RowMatrix P;
        RowMatrix W;
        RowMatrix G;
    };
}