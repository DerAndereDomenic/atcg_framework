//#define EIGEN_NO_CUDA
#include <Registration/CPDBackendCUDA.h>
#include <cutil.h>

namespace atcg
{
    namespace detail
    {
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

        __global__ void normalize(double* P, double* PX, double* PY, double* Z, uint32_t N, uint32_t M)
        {
            const size_t tid = cutil::globalThreadIndex();

            if(tid >= N*M)
            {
                return;
            }

            auto [n,m] = cutil::index1Dto2D(tid, N);

            P[tid] = P[tid]/(Z[n] + 1e-12);
            atomicAdd(&PX[n], P[tid]);
            atomicAdd(&PY[m], P[tid]);
        }
    }

    class CPDBackendCUDA::Impl
    {
    public:

        Impl() = default;

        ~Impl();

        double* devX,* devY,* devP,* devZ;
        double* devR, * devT;
        double* devPX,* devPY;
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
        cudaSafeCall(cudaFree(devPX));
        cudaSafeCall(cudaFree(devPY));
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
        cudaSafeCall(cudaMalloc((void**)&(impl->devPX), sizeof(double) * impl->N));
        cudaSafeCall(cudaMalloc((void**)&(impl->devPY), sizeof(double) * impl->M));
        

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
        cudaSafeCall(cudaMemset((void*)impl->devPX, 0, sizeof(double) * impl->N));
        cudaSafeCall(cudaMemset((void*)impl->devPY, 0, sizeof(double) * impl->M));

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

        detail::normalize<<<config.blocks, config.threads>>>(impl->devP,
                                                             impl->devPX,
                                                             impl->devPY,
                                                             impl->devZ,
                                                             impl->N,
                                                             impl->M);
        cutil::syncStream();

        cudaSafeCall(cudaMemcpy((void*)&P(0), (void*)(impl->devP), sizeof(double) * impl->N * impl->M, cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy((void*)&PX(0), (void*)(impl->devPX), sizeof(double) * impl->N, cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy((void*)&PY(0), (void*)(impl->devPY), sizeof(double) * impl->M, cudaMemcpyDeviceToHost));
    }
}