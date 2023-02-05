#include <Registration/NonRigidCPD.h>

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

#include <iostream>

namespace atcg
{
NonRigidCoherentPointDrift::NonRigidCoherentPointDrift(const std::shared_ptr<PointCloud>& source,
                                                       const std::shared_ptr<PointCloud>& target,
                                                       const double& w)
    : Registration::Registration(source, target)
    , w(w)
{
}

NonRigidCoherentPointDrift::~NonRigidCoherentPointDrift() {}

void NonRigidCoherentPointDrift::solve(const uint32_t& maxN, const float& tol)
{
    double var     = initialize();
    double old_var = 0.0;
    uint32_t n     = 0;
    while(n < maxN && std::abs(old_var - var) > tol)
    {
        ++n;
        std::cout << "Iteration: " << n << "\n";

        P                  = RowMatrix::Zero(M, N);
        Eigen::VectorXd PX = Eigen::VectorXd::Zero(N);
        Eigen::VectorXd PY = Eigen::VectorXd::Zero(M);

        estimate(PX, PY, var);

        old_var = var;
        var     = maximize(PY, PY, var);
        std::cout << var << "\n";
    }
}

double NonRigidCoherentPointDrift::initialize()
{
    G = RowMatrix(M, M);
    W = RowMatrix::Zero(M, 3);

    double var = 0.0;
    for(size_t m = 0; m < M; ++m)
    {
        for(size_t n = 0; n < N; ++n)
        {
            Eigen::Vector3d d = (X.block<1, 3>(n, 0) - Y.block<1, 3>(m, 0));
            var += d.dot(d) / static_cast<double>(3 * N * M);
        }

        for(size_t mm = m; mm < M; ++mm)
        {
            Eigen::Vector3d d = (Y.block<1, 3>(mm, 0) - Y.block<1, 3>(m, 0));
            double weight     = std::exp(-0.5 / (beta * beta) * d.dot(d));
            G(m, mm)          = weight;
            G(mm, m)          = weight;
        }
    }
    return var;
}

void NonRigidCoherentPointDrift::estimate(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double var)
{
    std::cout << "E-Step\n";
    double bias = std::pow(2.0 * glm::pi<double>() * var, 3.0 / 2.0) * w / (1.0 - w) * static_cast<double>(M) /
                  static_cast<double>(N);

    direct_optimization(PX, PY, bias, var);
}

void NonRigidCoherentPointDrift::direct_optimization(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double bias, double var)
{
    RowMatrix T = Y + G * W;
    for(size_t n = 0; n < N; ++n)
    {
        double Z = bias;
        for(size_t m = 0; m < M; ++m)
        {
            Eigen::Vector3d YV = T.block<1, 3>(m, 0);
            // YV += G.row(m) * W;

            P(m, n) = Pmn(X.block<1, 3>(n, 0), YV, var);
            Z += P(m, n);
        }

        for(size_t m = 0; m < M; ++m)
        {
            P(m, n) /= Z;
            PX(n) += P(m, n);    // PT1
            PY(m) += P(m, n);    // P1
        }
    }
}

double NonRigidCoherentPointDrift::Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var)
{
    return std::exp(-0.5f / var * (x - y).dot(x - y));
}

double NonRigidCoherentPointDrift::Pmn(const double& L2S, double var)
{
    return std::exp(-0.5f / var * L2S);
}

double NonRigidCoherentPointDrift::maximize(Eigen::VectorXd& PX, Eigen::VectorXd& PY, double var)
{
    std::cout << "M-Step\n";

    RowMatrix LHS = G + lambda * var * PY.cwiseInverse().asDiagonal().toDenseMatrix();
    RowMatrix RHS = PY.cwiseInverse().asDiagonal() * P * X - Y;
    W             = LHS.partialPivLu().solve(RHS);

    double Np   = 1.0 / P.sum();
    RowMatrix T = Y + G * W;

    var = Np / 3 *
          ((X.transpose() * PX.asDiagonal() * X).trace() - 2.0 * ((P * X).transpose() * T).trace() +
           (T.transpose() * PY.asDiagonal() * T).trace());
    return var;
}

void NonRigidCoherentPointDrift::applyTransform(const std::shared_ptr<PointCloud>& cloud)
{
    RowMatrix T = Y + G * W;
    for(auto v_it = cloud->vertices_begin(); v_it != cloud->vertices_end(); ++v_it)
    {
        PointCloud::Point p_ = cloud->point(*v_it);
        auto p               = T.block<1, 3>(v_it->idx(), 0);
        cloud->set_point(*v_it, PointCloud::Point {p(0), p(1), p(2)});
    }
}
}    // namespace atcg