#pragma once

#include <Eigen/Sparse>

namespace atcg
{
    template<typename T>
    struct Laplacian
    {
        Eigen::SparseMatrix<T> S; //Stiffness matrix
        Eigen::SparseMatrix<T> M; //Mass matrix
    };
}