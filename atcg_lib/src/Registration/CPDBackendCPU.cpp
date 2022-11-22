#include <Registration/CPDBackendCPU.h>

namespace atcg
{
    class CPDBackendCPU::Impl
    {
    public:

        Impl() = default;

        ~Impl() {};

        RowMatrix X,Y;
        uint32_t N,M;

        double Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var);
    };

    double CPDBackendCPU::Impl::Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var)
    {
        return std::exp(-0.5f/var*(x-y).dot(x-y));
    }

    CPDBackendCPU::CPDBackendCPU(const RowMatrix& X, const RowMatrix& Y)
        :CPDBackend(X,Y)
    {
        impl = std::make_unique<Impl>();
        impl->X = X;
        impl->Y = Y;
        impl->N = X.rows();
        impl->M = Y.rows();
    }

    void CPDBackendCPU::estimate(const Transformation& transform,
                                 RowMatrix& P, 
                                 Eigen::VectorXd& PX, 
                                 Eigen::VectorXd& PY, 
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
                P(m,n) = impl->Pmn(impl->X.block<1,3>(n,0), YV, var);
                Z(n) += P(m,n);
            }
        }

        for(size_t m = 0; m < impl->M; ++m)
        {
            for(size_t n = 0; n < impl->N; ++n)
            {
                P(m,n) /= Z(n);
                PX(n) += P(m,n); //PT1
                PY(m) += P(m,n); //P1
            }
        }
    }
}