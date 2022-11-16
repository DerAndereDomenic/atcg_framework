#include <Registration/CPD.h>

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

namespace atcg
{
    CoherentPointDrift::CoherentPointDrift(const std::shared_ptr<PointCloud>& source, const std::shared_ptr<PointCloud>& target, const double& w)
        :Registration::Registration(source, target), w(w)
    {

    }

    CoherentPointDrift::~CoherentPointDrift()
    {

    }

    void CoherentPointDrift::solve(const uint32_t& maxN, const float& tol = 0.01f)
    {
        double var = initialize();
        double old_var = 0.0;
        uint32_t n;
        while(n <= maxN && std::abs(old_var - var) > tol)
        {
            ++n;

            P = RowMatrix::Zero(M,N);
            Eigen::Vector3d PX = Eigen::Vector3d::Zero(N);
            Eigen::Vector3d PY = Eigen::Vector3d::Zero(M);

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

    void CoherentPointDrift::estimate(Eigen::Vector3d& PX, Eigen::Vector3d& PY, double var)
    {
        double bias = std::pow(2.0 * glm::pi<double>(), 3.0/2.0) * w / (1.0 - w) * static_cast<double>(M) / static_cast<double>(N);

        direct_optimization(PX, PY, bias, var);
    }

    void CoherentPointDrift::direct_optimization(Eigen::Vector3d& PX, Eigen::Vector3d& PY, double bias, double var)
    {
        for(size_t m = 0; m < M; ++m)
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
                PY(n) += P(m,n);
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

    double CoherentPointDrift::maximize(Eigen::Vector3d& PX, Eigen::Vector3d& PY)
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
}