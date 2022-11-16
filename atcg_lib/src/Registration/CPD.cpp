#include <Registration/CPD.h>

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

#include <iostream>

namespace atcg
{
    CoherentPointDrift::CoherentPointDrift(const std::shared_ptr<PointCloud>& source, const std::shared_ptr<PointCloud>& target, const double& w)
        :Registration::Registration(source, target), w(w)
    {

    }

    CoherentPointDrift::~CoherentPointDrift()
    {

    }

    void CoherentPointDrift::solve(const uint32_t& maxN, const float& tol)
    {
        double var = initialize();
        double old_var = 0.0;
        uint32_t n = 0;
        while(n < maxN && std::abs(old_var - var) > tol)
        {
            ++n;
            std::cout << "Iteration: " << n << "\n";

            P = RowMatrix::Zero(M,N);
            Eigen::VectorXd PX = Eigen::VectorXd::Zero(N);
            Eigen::VectorXd PY = Eigen::VectorXd::Zero(M);

            estimate(PX, PY, var);

            old_var = var;
            var = maximize(PY, PY);
        }
    }

    double CoherentPointDrift::initialize()
    {
        double var = 0.0;
        for(size_t n = 0; n < N; ++n)
        {
            for(size_t m = 0; m < M; ++m)
            {
                Eigen::Vector3d d = (X.block<1,3>(n, 0) - Y.block<1, 3>(m, 0));
                var += d.dot(d)/static_cast<double>(3*N*M);
            }
        }
        return var;
    }

    void CoherentPointDrift::estimate(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double var)
    {
        double bias = std::pow(2.0 * glm::pi<double>() * var, 3.0/2.0) * w / (1.0 - w) * static_cast<double>(M) / static_cast<double>(N);

        direct_optimization(PX, PY, bias, var);
    }

    void CoherentPointDrift::direct_optimization(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double bias, double var)
    {
        /*for(size_t m = 0; m < M; ++m)
        {
            double Z = bias;

            Eigen::Vector3d YV = Y.block<1,3>(m, 0);
            YV = s * R * YV + t;

            for(size_t n = 0; n < N; ++n)
            {
                P(m,n) = Pmn(X.block<1,3>(n, 0), YV, var);
                Z += P(m, n);
            }

            for(size_t n = 0; n < N; ++n)
            {
                P(m,n) /= Z;
                PX(n) += P(m,n);
                PY(m) += P(m,n);
            }
        }*/

        for(size_t n = 0; n < N; ++n)
        {
            double Z = bias;
            for(size_t m = 0; m < M; ++m)
            {
                Eigen::Vector3d YV = Y.block<1,3>(m, 0);
                YV = s * R * YV + t;

                P(m,n) = Pmn(X.block<1,3>(n,0), YV, var);
                Z += P(m,n);
            }

            for(size_t m = 0; m < M; ++m)
            {
                P(m,n) /= Z;
                PX(n) += P(m,n);
                PY(m) += P(m,n);
            }
        }
    }

    double CoherentPointDrift::Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var)
    {
        return std::exp(-0.5f/var*(x-y).dot(x-y));
    }

    double CoherentPointDrift::Pmn(const double& L2S, double var)
    {
        return std::exp(-0.5f/var*L2S);
    }

    double CoherentPointDrift::maximize(Eigen::VectorXd& PX, Eigen::VectorXd& PY)
    {
        double Np = 1.0/P.sum();
        Eigen::Vector3d uX = X.transpose()*PX*Np;
        Eigen::Vector3d uY = Y.transpose()*PY*Np;

        RowMatrix XC = X.rowwise() - uX.transpose();
        RowMatrix YC = Y.rowwise() - uY.transpose();

        RowMatrix A = XC.transpose() * P.transpose() * YC;

        svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        RowMatrix U = svd.matrixU();
        RowMatrix V = svd.matrixV();

        RowMatrix C = RowMatrix::Identity(3, 3);
        C(2, 2) = (U*V.transpose()).determinant();

        R = U*C*V.transpose();
        s = (A.transpose()*R).trace()/(YC.transpose()*PY.asDiagonal()*YC).trace();

        double var = Np/3*((XC.transpose()*PX.asDiagonal()*XC).trace() - s*(A.transpose()*R).trace());

        t = uX - s*R*uY;

        return var;
    }

    void CoherentPointDrift::applyTransform(const std::shared_ptr<PointCloud>& cloud)
    {
        for(auto v_it = cloud->vertices_begin(); v_it != cloud->vertices_end(); ++v_it)
        {
            PointCloud::Point p_ = cloud->point(*v_it);
            Eigen::Vector3d p;
            p(0) = p_[0];
            p(1) = p_[1];
            p(2) = p_[2];
            p = s * R * p + t;
            cloud->set_point(*v_it, PointCloud::Point{p(0), p(1), p(2)}); 
        }
    }
}