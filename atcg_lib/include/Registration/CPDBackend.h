#pragma once

#include <Math/Utils.h>
#include <Registration/Transformation.h>

namespace atcg
{
    class CPDBackend
    {
    public:
        CPDBackend(const RowMatrix& X, const RowMatrix& Y){}

        virtual ~CPDBackend() {}

        virtual void estimate(const Transformation& transform,
                              RowMatrix& P, 
                              Eigen::VectorXd& PX, 
                              Eigen::VectorXd& PY, 
                              double bias, 
                              double var) = 0;
    };
}