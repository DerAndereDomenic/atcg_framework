#include <Registration/CPDBackendCPU.h>

namespace atcg
{
    class CPDBackendCPU::Impl
    {
    public:

        Impl() = default;

        ~Impl() {};

        RowMatrix X,Y;
        RowMatrix P;
        uint32_t N,M;

        double Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var);
    };

    double CPDBackendCPU::Impl::Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var)
    {
        return std::exp(-0.5f/var*(x-y).dot(x-y));
    }

    CPDBackendCPU::CPDBackendCPU( RowMatrix& X,  RowMatrix& Y)
        :CPDBackend(X,Y)
    {
        impl = std::make_unique<Impl>();
        impl->X = X;
        impl->Y = Y;
        impl->N = X.rows();
        impl->M = Y.rows();
        impl->P = RowMatrix::Zero(impl->M,impl->N);
    }

    void CPDBackendCPU::estimate(const Transformation& transform,
                                 Eigen::VectorXd& PX, 
                                 Eigen::VectorXd& PY, 
                                 double& Np,
                                 double bias, 
                                 double var)
    {
        Eigen::VectorXd Z = Eigen::VectorXd::Constant(impl->N, bias);
        for(size_t m = 0; m < impl->M; ++m)
        {
            Eigen::Vector3d YV = impl->Y.block<1,3>(m, 0);
            YV = transform.s*transform.R*YV+transform.t;

            for(size_t n = 0; n < impl->N; ++n)
            {
                impl->P(m,n) = impl->Pmn(impl->X.block<1,3>(n,0), YV, var);
                Z(n) += impl->P(m,n);
            }
        }

        for(size_t m = 0; m < impl->M; ++m)
        {
            for(size_t n = 0; n < impl->N; ++n)
            {
                impl->P(m,n) = impl->P(m,n)/(Z(n) + 1e-12);
                PX(n) += impl->P(m,n); //PT1
                PY(m) += impl->P(m,n); //P1
            }
        }

        Np = 1.0/impl->P.sum();
    }

    void CPDBackendCPU::maximize(const RowMatrix& XC,
                                 const RowMatrix& YC,
                                RowMatrix& A)
    {
        A = XC.transpose() * impl->P.transpose() * YC;
    }
}