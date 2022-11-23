#pragma once

#include <Registration/CPDBackend.h>
#include <memory>

namespace atcg
{
    class CPDBackendCUDA : public CPDBackend
    {
    public:
        CPDBackendCUDA( RowMatrix& X,  RowMatrix& Y);

        virtual void estimate(const Transformation& transform,
                              RowMatrix& P, 
                              Eigen::VectorXd& PX, 
                              Eigen::VectorXd& PY, 
                              double bias, 
                              double var) override;
    private:
        class Impl;
        std::unique_ptr<Impl> impl;
    };
}