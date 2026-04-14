#pragma once

#include <Core/glm.h>
#include <Core/Platform.h>

namespace atcg
{
struct mat6
{
    mat6() = default;

    // scaled identity constructor
    ATCG_HOST_DEVICE
    mat6(float v) : m00(v), m01(0), m10(0), m11(v) {}

    ATCG_HOST_DEVICE
    mat6(const glm::mat3& m00, const glm::mat3& m01, const glm::mat3& m10, const glm::mat3& m11)
        : m00(m00),
          m01(m01),
          m10(m10),
          m11(m11)
    {
    }

    glm::mat3 m00;
    glm::mat3 m01;
    glm::mat3 m10;
    glm::mat3 m11;
};

struct mat6x3
{
    mat6x3() = default;

    ATCG_HOST_DEVICE
    mat6x3(float v) : m00(v), m01(0) {}

    ATCG_HOST_DEVICE
    mat6x3(const glm::mat3& m00, const glm::mat3& m01) : m00(m00), m01(m01) {}

    glm::mat3 m00;
    glm::mat3 m01;
};

struct mat4x6
{
    mat4x6() = default;

    ATCG_HOST_DEVICE
    mat4x6(const glm::mat2x3& m00, const glm::mat2x3& m01, const glm::mat2x3& m10, const glm::mat2x3& m11)
        : m00(m00),
          m01(m01),
          m10(m10),
          m11(m11)
    {
    }

    glm::mat2x3 m00;
    glm::mat2x3 m01;
    glm::mat2x3 m10;
    glm::mat2x3 m11;
};

struct mat6x4
{
    mat6x4() = default;

    ATCG_HOST_DEVICE
    mat6x4(const glm::mat3x2& m00, const glm::mat3x2& m01, const glm::mat3x2& m10, const glm::mat3x2& m11)
        : m00(m00),
          m01(m01),
          m10(m10),
          m11(m11)
    {
    }

    glm::mat3x2 m00;
    glm::mat3x2 m01;
    glm::mat3x2 m10;
    glm::mat3x2 m11;
};

ATCG_HOST_DEVICE ATCG_INLINE mat6 operator+(const mat6& A, const mat6& B)
{
    return mat6(A.m00 + B.m00, A.m01 + B.m01, A.m10 + B.m10, A.m11 + B.m11);
}

ATCG_HOST_DEVICE ATCG_INLINE void operator+=(mat6& A, const mat6& B)
{
    A = A + B;
}

ATCG_HOST_DEVICE ATCG_INLINE mat6 operator*(const mat6& A, const mat6& B)
{
    return mat6(A.m00 * B.m00 + A.m01 * B.m10,
                A.m00 * B.m01 + A.m01 * B.m11,
                A.m10 * B.m00 + A.m11 * B.m10,
                A.m10 * B.m01 + A.m11 * B.m11);
}

ATCG_HOST_DEVICE ATCG_INLINE mat6x3 operator+(const mat6x3& A, const mat6x3& B)
{
    return mat6x3(A.m00 + B.m00, A.m01 + B.m01);
}

ATCG_HOST_DEVICE ATCG_INLINE void operator+=(mat6x3& A, const mat6x3& B)
{
    A.m00 = A.m00 + B.m00;
    A.m01 = A.m01 + B.m01;
}

ATCG_HOST_DEVICE ATCG_INLINE void operator-=(mat6x3& A, const mat6x3& B)
{
    A.m00 = A.m00 - B.m00;
    A.m01 = A.m01 - B.m01;
}

ATCG_HOST_DEVICE ATCG_INLINE mat6x3 operator*(const mat6x3& A, const mat6& B)
{
    return mat6x3(A.m00 * B.m00 + A.m01 * B.m10, A.m00 * B.m01 + A.m01 * B.m11);
}

ATCG_HOST_DEVICE ATCG_INLINE mat6x3 operator*(const glm::mat3& A, const mat6x3& B)
{
    return mat6x3(A * B.m00, A * B.m01);
}

ATCG_HOST_DEVICE ATCG_INLINE mat4x6 operator*(const mat6& A, const mat4x6& B)
{
    return mat4x6(A.m00 * B.m00 + A.m01 * B.m10,
                  A.m00 * B.m01 + A.m01 * B.m11,
                  A.m10 * B.m00 + A.m11 * B.m10,
                  A.m10 * B.m01 + A.m11 * B.m11);
}

ATCG_HOST_DEVICE ATCG_INLINE glm::mat4 operator*(const mat6x4& A, const mat4x6& B)
{
    glm::mat2 m00 = A.m00 * B.m00 + A.m01 * B.m10;
    glm::mat2 m01 = A.m00 * B.m01 + A.m01 * B.m11;
    glm::mat2 m10 = A.m10 * B.m00 + A.m11 * B.m10;
    glm::mat2 m11 = A.m10 * B.m01 + A.m11 * B.m11;
    return glm::mat4(glm::vec4(m00[0], m10[0]),
                     glm::vec4(m00[1], m10[1]),
                     glm::vec4(m01[0], m11[0]),
                     glm::vec4(m01[1], m11[1]));
}

ATCG_HOST_DEVICE ATCG_INLINE glm::mat4x3 operator*(const mat6x3& A, const mat4x6& B)
{
    glm::mat2x3 m00 = A.m00 * B.m00 + A.m01 * B.m10;
    glm::mat2x3 m01 = A.m00 * B.m01 + A.m01 * B.m11;

    return glm::mat4x3(m00[0], m00[1], m01[0], m01[1]);
}

ATCG_HOST_DEVICE ATCG_INLINE mat6x4 transpose(const mat4x6& A)
{
    return mat6x4(glm::transpose(A.m00), glm::transpose(A.m10), glm::transpose(A.m01), glm::transpose(A.m11));
}
}    // namespace atcg