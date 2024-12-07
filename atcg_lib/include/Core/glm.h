#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_CTOR_INIT
#define GLM_ENABLE_EXPERIMENTAL
#include <Core/Platform.h>
#include <ostream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_cross_product.hpp>
#include <glm/gtx/vec_swizzle.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/exterior_product.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext/scalar_constants.hpp>

//
#ifndef __CUDACC__

namespace detail
{
template<glm::length_t N, typename T>
struct dispatch_print_vector
{
    static ATCG_INLINE std::ostream& apply(std::ostream& os, const glm::vec<N, T>& vector)
    {
        os << "[]";
        return os;
    }
};

template<typename T>
struct dispatch_print_vector<1, T>
{
    static ATCG_INLINE std::ostream& apply(std::ostream& os, const glm::vec<1, T>& vector)
    {
        os << "[" << vector.x << "]";
        return os;
    }
};
template<typename T>
struct dispatch_print_vector<2, T>
{
    static ATCG_INLINE std::ostream& apply(std::ostream& os, const glm::vec<2, T>& vector)
    {
        os << "[" << vector.x << ", " << vector.y << "]";
        return os;
    }
};

template<typename T>
struct dispatch_print_vector<3, T>
{
    static ATCG_INLINE std::ostream& apply(std::ostream& os, const glm::vec<3, T>& vector)
    {
        os << "[" << vector.x << ", " << vector.y << ", " << vector.z << "]";
        return os;
    }
};
template<typename T>
struct dispatch_print_vector<4, T>
{
    static ATCG_INLINE std::ostream& apply(std::ostream& os, const glm::vec<4, T>& vector)
    {
        os << "[" << vector.x << ", " << vector.y << ", " << vector.z << ", " << vector.w << "]";
        return os;
    }
};

template<glm::length_t N, glm::length_t M, typename T>
struct dispatch_print_matrix
{
    static ATCG_INLINE std::ostream& apply(std::ostream& os, const glm::mat<N, M, T>& mat)
    {
        os << "[";
        dispatch_print_vector<M, T>::apply(os, mat[0]);
        os << ",\n";
        for(glm::length_t i = 1; i < N - 1; ++i)
        {
            os << " ";
            dispatch_print_vector<M, T>::apply(os, mat[i]);
            os << ",\n";
        }
        os << " ";
        dispatch_print_vector<M, T>::apply(os, mat[N - 1]);
        os << "]";
        return os;
    }
};
}    // namespace detail

template<glm::length_t N, typename T>
ATCG_INLINE std::ostream& operator<<(std::ostream& os, const glm::vec<N, T>& vector)
{
    return detail::dispatch_print_vector<N, T>::apply(os, vector);
}

template<glm::length_t N, glm::length_t M, typename T>
ATCG_INLINE std::ostream& operator<<(std::ostream& os, const glm::mat<N, M, T>& mat)
{
    return detail::dispatch_print_matrix<N, M, T>::apply(os, mat);
}

#endif