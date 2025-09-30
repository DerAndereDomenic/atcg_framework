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

#include <type_traits>

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

#ifdef ATCG_CUDA_BACKEND

    #include <Core/Platform.h>
    #include "cuda_runtime.h"

    #pragma region is_glm_vec

template<typename T>
struct is_glm_vec : std::false_type
{
};

template<int N, typename Q, glm::qualifier P>
struct is_glm_vec<glm::vec<N, Q, P>> : std::true_type
{
};

    #pragma endregion is_glm_vec

    #pragma region vec_traits

template<typename T>
struct vec_traits
{
    static_assert(true, "Not a vector type");
};

template<int N, typename Q, glm::qualifier P>
struct vec_traits<glm::vec<N, Q, P>>
{
    using vector_type        = glm::vec<N, Q, P>;
    using scalar_type        = Q;
    static constexpr int dim = N;
};

// ---- char ----
template<>
struct vec_traits<char1>
{
    using vector_type        = char1;
    using scalar_type        = signed char;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<char2>
{
    using vector_type        = char2;
    using scalar_type        = signed char;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<char3>
{
    using vector_type        = char3;
    using scalar_type        = signed char;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<char4>
{
    using vector_type        = char4;
    using scalar_type        = signed char;
    static constexpr int dim = 4;
};

// ---- unsigned char ----
template<>
struct vec_traits<uchar1>
{
    using vector_type        = uchar1;
    using scalar_type        = unsigned char;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<uchar2>
{
    using vector_type        = uchar2;
    using scalar_type        = unsigned char;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<uchar3>
{
    using vector_type        = uchar3;
    using scalar_type        = unsigned char;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<uchar4>
{
    using vector_type        = uchar4;
    using scalar_type        = unsigned char;
    static constexpr int dim = 4;
};

// ---- short ----
template<>
struct vec_traits<short1>
{
    using vector_type        = short1;
    using scalar_type        = short;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<short2>
{
    using vector_type        = short2;
    using scalar_type        = short;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<short3>
{
    using vector_type        = short3;
    using scalar_type        = short;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<short4>
{
    using vector_type        = short4;
    using scalar_type        = short;
    static constexpr int dim = 4;
};

// ---- unsigned short ----
template<>
struct vec_traits<ushort1>
{
    using vector_type        = ushort1;
    using scalar_type        = unsigned short;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<ushort2>
{
    using vector_type        = ushort2;
    using scalar_type        = unsigned short;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<ushort3>
{
    using vector_type        = ushort3;
    using scalar_type        = unsigned short;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<ushort4>
{
    using vector_type        = ushort4;
    using scalar_type        = unsigned short;
    static constexpr int dim = 4;
};

// ---- int ----
template<>
struct vec_traits<int1>
{
    using vector_type        = int1;
    using scalar_type        = int;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<int2>
{
    using vector_type        = int2;
    using scalar_type        = int;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<int3>
{
    using vector_type        = int3;
    using scalar_type        = int;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<int4>
{
    using vector_type        = int4;
    using scalar_type        = int;
    static constexpr int dim = 4;
};

// ---- unsigned int ----
template<>
struct vec_traits<uint1>
{
    using vector_type        = uint1;
    using scalar_type        = unsigned int;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<uint2>
{
    using vector_type        = uint2;
    using scalar_type        = unsigned int;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<uint3>
{
    using vector_type        = uint3;
    using scalar_type        = unsigned int;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<uint4>
{
    using vector_type        = uint4;
    using scalar_type        = unsigned int;
    static constexpr int dim = 4;
};

// ---- long long ----
template<>
struct vec_traits<longlong1>
{
    using vector_type        = longlong1;
    using scalar_type        = long long;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<longlong2>
{
    using vector_type        = longlong2;
    using scalar_type        = long long;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<longlong3>
{
    using vector_type        = longlong3;
    using scalar_type        = long long;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<longlong4>
{
    using vector_type        = longlong4;
    using scalar_type        = long long;
    static constexpr int dim = 4;
};

// ---- unsigned long long ----
template<>
struct vec_traits<ulonglong1>
{
    using vector_type        = ulonglong1;
    using scalar_type        = unsigned long long;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<ulonglong2>
{
    using vector_type        = ulonglong2;
    using scalar_type        = unsigned long long;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<ulonglong3>
{
    using vector_type        = ulonglong3;
    using scalar_type        = unsigned long long;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<ulonglong4>
{
    using vector_type        = ulonglong4;
    using scalar_type        = unsigned long long;
    static constexpr int dim = 4;
};

// ---- float ----
template<>
struct vec_traits<float1>
{
    using vector_type        = float1;
    using scalar_type        = float;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<float2>
{
    using vector_type        = float2;
    using scalar_type        = float;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<float3>
{
    using vector_type        = float3;
    using scalar_type        = float;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<float4>
{
    using vector_type        = float4;
    using scalar_type        = float;
    static constexpr int dim = 4;
};

// ---- double ----
template<>
struct vec_traits<double1>
{
    using vector_type        = double1;
    using scalar_type        = double;
    static constexpr int dim = 1;
};

template<>
struct vec_traits<double2>
{
    using vector_type        = double2;
    using scalar_type        = double;
    static constexpr int dim = 2;
};

template<>
struct vec_traits<double3>
{
    using vector_type        = double3;
    using scalar_type        = double;
    static constexpr int dim = 3;
};

template<>
struct vec_traits<double4>
{
    using vector_type        = double4;
    using scalar_type        = double;
    static constexpr int dim = 4;
};
    #pragma endregion vec_traits

/**
 * @author Tom Kneiphof
 */
    #pragma region cuda_type

template<int Dim, typename Scalar>
struct cuda_type
{
    static_assert(sizeof(Scalar) == 0, "Not a cuda type (i.e. int3 or float4)!");
};

template<>
struct cuda_type<1, glm::i8>
{
    typedef char1 vector_type;
    typedef char scalar_type;
};
template<>
struct cuda_type<2, glm::i8>
{
    typedef char2 vector_type;
    typedef char scalar_type;
};
template<>
struct cuda_type<3, glm::i8>
{
    typedef char3 vector_type;
    typedef char scalar_type;
};
template<>
struct cuda_type<4, glm::i8>
{
    typedef char4 vector_type;
    typedef char scalar_type;
};
template<>
struct cuda_type<1, glm::u8>
{
    typedef uchar1 vector_type;
    typedef unsigned char scalar_type;
};
template<>
struct cuda_type<2, glm::u8>
{
    typedef uchar2 vector_type;
    typedef unsigned char scalar_type;
};
template<>
struct cuda_type<3, glm::u8>
{
    typedef uchar3 vector_type;
    typedef unsigned char scalar_type;
};
template<>
struct cuda_type<4, glm::u8>
{
    typedef uchar4 vector_type;
    typedef unsigned char scalar_type;
};
template<>
struct cuda_type<1, glm::i16>
{
    typedef short1 vector_type;
    typedef short scalar_type;
};
template<>
struct cuda_type<2, glm::i16>
{
    typedef short2 vector_type;
    typedef short scalar_type;
};
template<>
struct cuda_type<3, glm::i16>
{
    typedef short3 vector_type;
    typedef short scalar_type;
};
template<>
struct cuda_type<4, glm::i16>
{
    typedef short4 vector_type;
    typedef short scalar_type;
};
template<>
struct cuda_type<1, glm::u16>
{
    typedef ushort1 vector_type;
    typedef unsigned short scalar_type;
};
template<>
struct cuda_type<2, glm::u16>
{
    typedef ushort2 vector_type;
    typedef unsigned short scalar_type;
};
template<>
struct cuda_type<3, glm::u16>
{
    typedef ushort3 vector_type;
    typedef unsigned short scalar_type;
};
template<>
struct cuda_type<4, glm::u16>
{
    typedef ushort4 vector_type;
    typedef unsigned short scalar_type;
};
template<>
struct cuda_type<1, glm::i32>
{
    typedef int1 vector_type;
    typedef int scalar_type;
};
template<>
struct cuda_type<2, glm::i32>
{
    typedef int2 vector_type;
    typedef int scalar_type;
};
template<>
struct cuda_type<3, glm::i32>
{
    typedef int3 vector_type;
    typedef int scalar_type;
};
template<>
struct cuda_type<4, glm::i32>
{
    typedef int4 vector_type;
    typedef int scalar_type;
};
template<>
struct cuda_type<1, glm::u32>
{
    typedef uint1 vector_type;
    typedef unsigned int scalar_type;
};
template<>
struct cuda_type<2, glm::u32>
{
    typedef uint2 vector_type;
    typedef unsigned int scalar_type;
};
template<>
struct cuda_type<3, glm::u32>
{
    typedef uint3 vector_type;
    typedef unsigned int scalar_type;
};
template<>
struct cuda_type<4, glm::u32>
{
    typedef uint4 vector_type;
    typedef unsigned int scalar_type;
};
template<>
struct cuda_type<1, glm::i64>
{
    typedef longlong1 vector_type;
    typedef long long scalar_type;
};
template<>
struct cuda_type<2, glm::i64>
{
    typedef longlong2 vector_type;
    typedef long long scalar_type;
};
template<>
struct cuda_type<3, glm::i64>
{
    typedef longlong3 vector_type;
    typedef long long scalar_type;
};
template<>
struct cuda_type<4, glm::i64>
{
    typedef longlong4 vector_type;
    typedef long long scalar_type;
};
template<>
struct cuda_type<1, glm::u64>
{
    typedef ulonglong1 vector_type;
    typedef unsigned long long scalar_type;
};
template<>
struct cuda_type<2, glm::u64>
{
    typedef ulonglong2 vector_type;
    typedef unsigned long long scalar_type;
};
template<>
struct cuda_type<3, glm::u64>
{
    typedef ulonglong3 vector_type;
    typedef unsigned long long scalar_type;
};
template<>
struct cuda_type<4, glm::u64>
{
    typedef ulonglong4 vector_type;
    typedef unsigned long long scalar_type;
};
template<>
struct cuda_type<1, glm::f32>
{
    typedef float1 vector_type;
    typedef float scalar_type;
};
template<>
struct cuda_type<2, glm::f32>
{
    typedef float2 vector_type;
    typedef float scalar_type;
};
template<>
struct cuda_type<3, glm::f32>
{
    typedef float3 vector_type;
    typedef float scalar_type;
};
template<>
struct cuda_type<4, glm::f32>
{
    typedef float4 vector_type;
    typedef float scalar_type;
};
template<>
struct cuda_type<1, glm::f64>
{
    typedef double1 vector_type;
    typedef double scalar_type;
};
template<>
struct cuda_type<2, glm::f64>
{
    typedef double2 vector_type;
    typedef double scalar_type;
};
template<>
struct cuda_type<3, glm::f64>
{
    typedef double3 vector_type;
    typedef double scalar_type;
};
template<>
struct cuda_type<4, glm::f64>
{
    typedef double4 vector_type;
    typedef double scalar_type;
};

    #pragma endregion cuda_type

    #pragma region make_cuda_type

template<int N, typename T>
struct make_cuda_type
{
    static_assert(sizeof(T) == 0, "Not a cuda type (i.e. int3 or float4)!");
};

template<>
struct make_cuda_type<1, glm::i8> : public cuda_type<1, glm::i8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_char1(x); }
};
template<>
struct make_cuda_type<2, glm::i8> : public cuda_type<2, glm::i8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_char2(x, y); }
};
template<>
struct make_cuda_type<3, glm::i8> : public cuda_type<3, glm::i8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_char3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::i8> : public cuda_type<4, glm::i8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_char4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::u8> : public cuda_type<1, glm::u8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_uchar1(x); }
};
template<>
struct make_cuda_type<2, glm::u8> : public cuda_type<2, glm::u8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_uchar2(x, y); }
};
template<>
struct make_cuda_type<3, glm::u8> : public cuda_type<3, glm::u8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_uchar3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::u8> : public cuda_type<4, glm::u8>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_uchar4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::i16> : public cuda_type<1, glm::i16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_short1(x); }
};
template<>
struct make_cuda_type<2, glm::i16> : public cuda_type<2, glm::i16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_short2(x, y); }
};
template<>
struct make_cuda_type<3, glm::i16> : public cuda_type<3, glm::i16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_short3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::i16> : public cuda_type<4, glm::i16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_short4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::u16> : public cuda_type<1, glm::u16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_ushort1(x); }
};
template<>
struct make_cuda_type<2, glm::u16> : public cuda_type<2, glm::u16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y)
    {
        return make_ushort2(x, y);
    }
};
template<>
struct make_cuda_type<3, glm::u16> : public cuda_type<3, glm::u16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_ushort3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::u16> : public cuda_type<4, glm::u16>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_ushort4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::i32> : public cuda_type<1, glm::i32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_int1(x); }
};
template<>
struct make_cuda_type<2, glm::i32> : public cuda_type<2, glm::i32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_int2(x, y); }
};
template<>
struct make_cuda_type<3, glm::i32> : public cuda_type<3, glm::i32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_int3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::i32> : public cuda_type<4, glm::i32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_int4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::u32> : public cuda_type<1, glm::u32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_uint1(x); }
};
template<>
struct make_cuda_type<2, glm::u32> : public cuda_type<2, glm::u32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_uint2(x, y); }
};
template<>
struct make_cuda_type<3, glm::u32> : public cuda_type<3, glm::u32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_uint3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::u32> : public cuda_type<4, glm::u32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_uint4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::i64> : public cuda_type<1, glm::i64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_longlong1(x); }
};
template<>
struct make_cuda_type<2, glm::i64> : public cuda_type<2, glm::i64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y)
    {
        return make_longlong2(x, y);
    }
};
template<>
struct make_cuda_type<3, glm::i64> : public cuda_type<3, glm::i64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_longlong3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::i64> : public cuda_type<4, glm::i64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_longlong4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::u64> : public cuda_type<1, glm::u64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_ulonglong1(x); }
};
template<>
struct make_cuda_type<2, glm::u64> : public cuda_type<2, glm::u64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y)
    {
        return make_ulonglong2(x, y);
    }
};
template<>
struct make_cuda_type<3, glm::u64> : public cuda_type<3, glm::u64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_ulonglong3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::u64> : public cuda_type<4, glm::u64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_ulonglong4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::f32> : public cuda_type<1, glm::f32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_float1(x); }
};
template<>
struct make_cuda_type<2, glm::f32> : public cuda_type<2, glm::f32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y) { return make_float2(x, y); }
};
template<>
struct make_cuda_type<3, glm::f32> : public cuda_type<3, glm::f32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_float3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::f32> : public cuda_type<4, glm::f32>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_float4(x, y, z, w);
    }
};
template<>
struct make_cuda_type<1, glm::f64> : public cuda_type<1, glm::f64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x) { return make_double1(x); }
};
template<>
struct make_cuda_type<2, glm::f64> : public cuda_type<2, glm::f64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y)
    {
        return make_double2(x, y);
    }
};
template<>
struct make_cuda_type<3, glm::f64> : public cuda_type<3, glm::f64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z)
    {
        return make_double3(x, y, z);
    }
};
template<>
struct make_cuda_type<4, glm::f64> : public cuda_type<4, glm::f64>
{
    static __host__ __device__ ATCG_INLINE vector_type apply(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        return make_double4(x, y, z, w);
    }
};

    #pragma endregion make_cuda_type

    #pragma region glm2cuda

template<glm::length_t N, typename T, glm::qualifier Q>
struct glm2cuda_detail
{
    static_assert(sizeof(T) == 0, "Not implemented for this type!");
};

template<typename T, glm::qualifier Q>
struct glm2cuda_detail<1, T, Q>
{
    static constexpr int N = 1;
    static __host__ __device__ ATCG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v)
    {
        return make_cuda_type<N, T>::apply(v.x);
    };
};

template<typename T, glm::qualifier Q>
struct glm2cuda_detail<2, T, Q>
{
    static constexpr int N = 2;
    static __host__ __device__ ATCG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v)
    {
        return make_cuda_type<N, T>::apply(v.x, v.y);
    };
};

template<typename T, glm::qualifier Q>
struct glm2cuda_detail<3, T, Q>
{
    static constexpr int N = 3;
    static __host__ __device__ ATCG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v)
    {
        return make_cuda_type<N, T>::apply(v.x, v.y, v.z);
    };
};

template<typename T, glm::qualifier Q>
struct glm2cuda_detail<4, T, Q>
{
    static constexpr int N = 4;
    static __host__ __device__ ATCG_INLINE typename make_cuda_type<N, T>::vector_type apply(const glm::vec<N, T, Q>& v)
    {
        return make_cuda_type<N, T>::apply(v.x, v.y, v.z, v.w);
    };
};

template<glm::length_t N, typename T, glm::qualifier Q>
__host__ __device__ ATCG_INLINE typename cuda_type<N, T>::vector_type glm2cuda(const glm::vec<N, T, Q>& v)
{
    return glm2cuda_detail<N, T, Q>::apply(v);
}

    #pragma endregion glm2cuda

    #pragma region cuda2glm

template<int N, typename T, glm::qualifier Q>
struct cuda2glm_detail
{
    static_assert(sizeof(T) == 0, "Not implemented for type!");
};

template<typename T, glm::qualifier Q>
struct cuda2glm_detail<1, T, Q>
{
    typedef glm::vec<1, T, Q> result_type;
    static __host__ __device__ ATCG_INLINE result_type apply(const typename cuda_type<1, T>::vector_type& v)
    {
        return result_type(v.x);
    }
};

template<typename T, glm::qualifier Q>
struct cuda2glm_detail<2, T, Q>
{
    typedef glm::vec<2, T, Q> result_type;
    static __host__ __device__ ATCG_INLINE result_type apply(const typename cuda_type<2, T>::vector_type& v)
    {
        return result_type(v.x, v.y);
    }
};

template<typename T, glm::qualifier Q>
struct cuda2glm_detail<3, T, Q>
{
    typedef glm::vec<3, T, Q> result_type;
    static __host__ __device__ ATCG_INLINE result_type apply(const typename cuda_type<3, T>::vector_type& v)
    {
        return result_type(v.x, v.y, v.z);
    }
};

template<typename T, glm::qualifier Q>
struct cuda2glm_detail<4, T, Q>
{
    typedef glm::vec<4, T, Q> result_type;
    static __host__ __device__ ATCG_INLINE result_type apply(const typename cuda_type<4, T>::vector_type& v)
    {
        return result_type(v.x, v.y, v.z, v.w);
    }
};

template<typename CudaVectorType, glm::qualifier Q = glm::defaultp>
__host__ __device__ ATCG_INLINE typename cuda2glm_detail<vec_traits<CudaVectorType>::dim,
                                                         typename vec_traits<CudaVectorType>::scalar_type,
                                                         Q>::result_type
cuda2glm(const CudaVectorType& v)
{
    return cuda2glm_detail<vec_traits<CudaVectorType>::dim, typename vec_traits<CudaVectorType>::scalar_type, Q>::apply(
        v);
};

#endif