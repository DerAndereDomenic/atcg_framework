#pragma once

#include <Registration/CPDBackend.h>
#include <memory>

namespace atcg
{
    class CPDBackendCPU : public CPDBackend
    {
    public:
        CPDBackendCPU( RowMatrix& X,  RowMatrix& Y);

        virtual void estimate(const Transformation& transform,
                              Eigen::VectorXd& PX, 
                              Eigen::VectorXd& PY,
                              double& Np,
                              double bias, 
                              double var) override;

        virtual void maximize(const RowMatrix& XC,
                              const RowMatrix& YC,
                              RowMatrix& A) override;
    private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };
}