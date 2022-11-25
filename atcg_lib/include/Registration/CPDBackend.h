#pragma once

#include <Math/Utils.h>
#include <Registration/Transformation.h>

namespace atcg
{
    class CPDBackend
    {
    public:
        CPDBackend( RowMatrix& X,  RowMatrix& Y){}

        virtual ~CPDBackend() {}

        virtual void estimate(const Transformation& transform, 
                              Eigen::VectorXd& PX, 
                              Eigen::VectorXd& PY,
                              double& Np, 
                              double bias, 
                              double var) = 0;

        virtual void maximize(const RowMatrix& XC,
                              const RowMatrix& YC,
                              RowMatrix& A) = 0;
    };
}