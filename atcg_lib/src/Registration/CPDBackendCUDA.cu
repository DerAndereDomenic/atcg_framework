//#define EIGEN_NO_CUDA
#include <Registration/CPDBackendCUDA.h>
#include <cutil.h>

namespace atcg
{
    namespace detail
    {
        double Pmn(const Eigen::Vector3d& x, const Eigen::Vector3d& y, double var)
        {
            return std::exp(-0.5f/var*(x-y).dot(x-y));
        }

        __global__ void fillP(double* X, double* Y, double* P, double* R, double* t, double* Z, double s, double var, uint32_t N, uint32_t M)
        {
            const size_t tid = cutil::globalThreadIndex();

            if(tid >= M*N)
                return;

            auto [n,m] = cutil::index1Dto2D(tid, N);

            double* x = X + 3*n;
            double* y = Y + 3*m;

            double d[3];

            d[0] = s*(R[0 + 0*3] * y[0] + R[0 + 1*3] * y[1] + R[0 + 2*3] * y[2]) + t[0] - x[0];
            d[1] = s*(R[1 + 0*3] * y[0] + R[1 + 1*3] * y[1] + R[1 + 2*3] * y[2]) + t[1] - x[1];
            d[2] = s*(R[2 + 0*3] * y[0] + R[2 + 1*3] * y[1] + R[2 + 2*3] * y[2]) + t[2] - x[2];

            P[tid] = std::exp(-0.5f/var * (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]));

            atomicAdd(&Z[n], P[tid]);
        }
    }

    class CPDBackendCUDA::Impl
    {
    public:

        Impl() = default;

        ~Impl();

        double* devX,* devY,* devP,* devZ;
        double* devR, * devT;
        uint32_t N,M;
    };

    CPDBackendCUDA::Impl::~Impl()
    {
        cudaSafeCall(cudaFree(devX));
        cudaSafeCall(cudaFree(devY));
        cudaSafeCall(cudaFree(devP));
        cudaSafeCall(cudaFree(devR));
        cudaSafeCall(cudaFree(devT));
        cudaSafeCall(cudaFree(devZ));
    }
    
    CPDBackendCUDA::CPDBackendCUDA( RowMatrix& X,  RowMatrix& Y)
        :CPDBackend(X,Y)
    {
        impl = std::make_unique<Impl>();
        impl->N = X.rows();
        impl->M = Y.rows();
        cudaSafeCall(cudaMalloc((void**)&(impl->devX), sizeof(double) * impl->N * 3));
        cudaSafeCall(cudaMalloc((void**)&(impl->devY), sizeof(double) * impl->M * 3));
        cudaSafeCall(cudaMalloc((void**)&(impl->devP), sizeof(double) * impl->M * impl->N));
        cudaSafeCall(cudaMalloc((void**)&(impl->devZ), sizeof(double) * impl->N));
        

        cudaSafeCall(cudaMalloc((void**)&(impl->devR), sizeof(double) * 3 * 3));
        cudaSafeCall(cudaMalloc((void**)&(impl->devT), sizeof(double) * 3));

        cudaSafeCall(cudaMemcpy((void*)(impl->devX), (void*)&X(0), sizeof(double) * impl->N * 3, cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy((void*)(impl->devY), (void*)&Y(0), sizeof(double) * impl->M * 3, cudaMemcpyHostToDevice));
    }

    void CPDBackendCUDA::estimate(const Transformation& transform,
                                 RowMatrix& P, 
                                 Eigen::VectorXd& PX, 
                                 Eigen::VectorXd& PY, 
                                 double bias, 
                                 double var)
    {
        cudaSafeCall(cudaMemcpy((void*)(impl->devR), (void*)&transform.R(0), sizeof(double) * 3 * 3, cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy((void*)(impl->devT), (void*)&transform.t(0), sizeof(double) * 3, cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemset((void*)impl->devZ, bias, sizeof(double) * impl->N));

        cutil::KernelSize config = cutil::configureKernel(impl->N * impl->M);
        detail::fillP<<<config.blocks, config.threads>>>(impl->devX,
                                                         impl->devY,
                                                         impl->devP,
                                                         impl->devR,
                                                         impl->devT,
                                                         impl->devZ,
                                                         transform.s,
                                                         var,
                                                         impl->N,
                                                         impl->M);
        cutil::syncStream();

        cudaSafeCall(cudaMemcpy((void*)&P(0), (void*)(impl->devP), sizeof(double) * impl->N * impl->M, cudaMemcpyDeviceToHost));

        Eigen::VectorXd Z = Eigen::VectorXd::Zero(impl->N);
        cudaSafeCall(cudaMemcpy((void*)&Z(0), (void*)(impl->devZ), sizeof(double) * impl->N, cudaMemcpyDeviceToHost));
        /*for(size_t m = 0; m < impl->M; ++m)
        {
            for(size_t n = 0; n < impl->N; ++n)
            {
                Z(n) += P(m,n);
            }
        }*/

        for(size_t m = 0; m < impl->M; ++m)
        {
            for(size_t n = 0; n < impl->N; ++n)
            {
                P(m,n) = P(m,n)/Z(n);
                PX(n) += P(m,n); //PT1
                PY(m) += P(m,n); //P1
            }
        }
    }
}