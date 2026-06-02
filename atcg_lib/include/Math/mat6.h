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

struct mat3x6
{
    mat3x6() = default;

    ATCG_HOST_DEVICE
    mat3x6(float v) : m00(v), m10(0) {}

    ATCG_HOST_DEVICE
    mat3x6(const glm::mat3& m00, const glm::mat3& m10) : m00(m00), m10(m10) {}

    glm::mat3 m00;
    glm::mat3 m10;
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

struct vec6
{
    vec6() = default;

    ATCG_HOST_DEVICE
    vec6(const glm::vec3& a, const glm::vec3& b) : a(a), b(b) {}

    glm::vec3 a;
    glm::vec3 b;
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

ATCG_HOST_DEVICE ATCG_INLINE vec6 operator*(const mat6& A, const vec6& v)
{
    return vec6(A.m00 * v.a + A.m01 * v.b, A.m10 * v.a + A.m11 * v.b);
}

ATCG_HOST_DEVICE ATCG_INLINE mat6 inverse(const mat6& A)
{
    // Compute the inverse of a 6x6 matrix using block matrix inversion
    glm::mat3 A00_inv              = glm::inverse(A.m00);
    glm::mat3 Schur_complement     = A.m11 - A.m10 * A00_inv * A.m01;
    glm::mat3 Schur_complement_inv = glm::inverse(Schur_complement);

    mat6 A_inv;
    A_inv.m00 = A00_inv + A00_inv * A.m01 * Schur_complement_inv * A.m10 * A00_inv;
    A_inv.m01 = -A00_inv * A.m01 * Schur_complement_inv;
    A_inv.m10 = -Schur_complement_inv * A.m10 * A00_inv;
    A_inv.m11 = Schur_complement_inv;

    return A_inv;
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

ATCG_HOST_DEVICE ATCG_INLINE mat6x3 operator/(const mat6x3& A, const float s)
{
    return mat6x3(A.m00 / s, A.m01 / s);
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

ATCG_HOST_DEVICE ATCG_INLINE mat6x3 transpose(const mat3x6& A)
{
    return mat6x3(glm::transpose(A.m00), glm::transpose(A.m10));
}

ATCG_HOST_DEVICE ATCG_INLINE mat3x6 transpose(const mat6x3& A)
{
    return mat3x6(glm::transpose(A.m00), glm::transpose(A.m01));
}

ATCG_HOST_DEVICE ATCG_INLINE mat6 transpose(const mat6& A)
{
    return mat6(glm::transpose(A.m00), glm::transpose(A.m10), glm::transpose(A.m01), glm::transpose(A.m11));
}

ATCG_HOST_DEVICE ATCG_INLINE float mat6_get(const mat6& A, int row, int col)
{
    // Select block and local coordinates
    const glm::mat3& block = (row < 3) ? ((col < 3) ? A.m00 : A.m01) : ((col < 3) ? A.m10 : A.m11);
    return block[col % 3][row % 3];    // [col][row] = column-major
}

ATCG_HOST_DEVICE ATCG_INLINE float& mat6_ref(mat6& A, int row, int col)
{
    glm::mat3& block = (row < 3) ? ((col < 3) ? A.m00 : A.m01) : ((col < 3) ? A.m10 : A.m11);
    return block[col % 3][row % 3];
}

ATCG_HOST_DEVICE ATCG_INLINE float mat4_get(const glm::mat4& A, int row, int col)
{
    return A[col][row];    // [col][row] = column-major
}

ATCG_HOST_DEVICE ATCG_INLINE float& mat4_ref(glm::mat4& A, int row, int col)
{
    return A[col][row];
}

ATCG_HOST_DEVICE ATCG_INLINE float& vec6_ref(vec6& v, int i)
{
    return (i < 3) ? v.a[i] : v.b[i - 3];
}

ATCG_HOST_DEVICE ATCG_INLINE float vec6_get(const vec6& v, int i)
{
    return (i < 3) ? v.a[i] : v.b[i - 3];
}

ATCG_HOST_DEVICE ATCG_INLINE void jacobi_svd6(const mat6& A, mat6& U, float sigma[6], mat6& Vt)
{
    constexpr int N      = 6;
    constexpr int SWEEPS = 30;
    constexpr float EPS  = 1e-6f;

    mat6 B = A;
    mat6 V(1.f);

    for(int sweep = 0; sweep < SWEEPS; ++sweep)
    {
        float off = 0.f;

        for(int p = 0; p < N - 1; ++p)
            for(int q = p + 1; q < N; ++q)
            {
                float app = 0.f, aqq = 0.f, apq = 0.f;
                for(int i = 0; i < N; ++i)
                {
                    float bp = mat6_get(B, i, p);
                    float bq = mat6_get(B, i, q);
                    app += bp * bp;
                    aqq += bq * bq;
                    apq += bp * bq;
                }
                off += apq * apq;

                // Skip if already orthogonal or either column is (near) zero
                if(app < EPS * EPS || aqq < EPS * EPS) continue;
                if(fabsf(apq) < EPS * sqrtf(app * aqq)) continue;

                // Numerically stable symmetric Schur decomposition
                float tau = (aqq - app) / (2.f * apq);
                float t   = copysignf(1.f, tau) / (fabsf(tau) + sqrtf(1.f + tau * tau));
                float c   = 1.0f / sqrtf(1.f + t * t);
                float s   = c * t;

                for(int i = 0; i < N; ++i)
                {
                    float bp          = mat6_get(B, i, p);
                    float bq          = mat6_get(B, i, q);
                    mat6_ref(B, i, p) = c * bp - s * bq;
                    mat6_ref(B, i, q) = s * bp + c * bq;
                }
                for(int i = 0; i < N; ++i)
                {
                    float vp          = mat6_get(V, i, p);
                    float vq          = mat6_get(V, i, q);
                    mat6_ref(V, i, p) = c * vp - s * vq;
                    mat6_ref(V, i, q) = s * vp + c * vq;
                }
            }
        if(off < EPS * EPS) break;
    }

    for(int j = 0; j < N; ++j)
    {
        float norm = 0.f;
        for(int i = 0; i < N; ++i)
        {
            float b = mat6_get(B, i, j);
            norm += b * b;
        }
        norm     = sqrtf(norm);
        sigma[j] = norm;

        float inv = (norm > EPS) ? 1.f / norm : 0.f;
        for(int i = 0; i < N; ++i)
            mat6_ref(U, i, j) = mat6_get(B, i, j) * inv;
    }

    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            mat6_ref(Vt, i, j) = mat6_get(V, j, i);
}

ATCG_HOST_DEVICE ATCG_INLINE void jacobi_svd4(const glm::mat4& A, glm::mat4& U, float sigma[4], glm::mat4& Vt)
{
    constexpr int N      = 4;
    constexpr int SWEEPS = 30;
    constexpr float EPS  = 1e-6f;

    glm::mat4 B = A;
    glm::mat4 V(1.f);

    for(int sweep = 0; sweep < SWEEPS; ++sweep)
    {
        float off = 0.f;

        for(int p = 0; p < N - 1; ++p)
            for(int q = p + 1; q < N; ++q)
            {
                float app = 0.f, aqq = 0.f, apq = 0.f;
                for(int i = 0; i < N; ++i)
                {
                    float bp = mat4_get(B, i, p);
                    float bq = mat4_get(B, i, q);
                    app += bp * bp;
                    aqq += bq * bq;
                    apq += bp * bq;
                }
                off += apq * apq;

                // Skip if already orthogonal or either column is (near) zero
                if(app < EPS * EPS || aqq < EPS * EPS) continue;
                if(fabsf(apq) < EPS * sqrtf(app * aqq)) continue;

                // Numerically stable symmetric Schur decomposition
                float tau = (aqq - app) / (2.f * apq);
                float t   = copysignf(1.f, tau) / (fabsf(tau) + sqrtf(1.f + tau * tau));
                float c   = 1.0f / sqrtf(1.f + t * t);
                float s   = c * t;

                for(int i = 0; i < N; ++i)
                {
                    float bp          = mat4_get(B, i, p);
                    float bq          = mat4_get(B, i, q);
                    mat4_ref(B, i, p) = c * bp - s * bq;
                    mat4_ref(B, i, q) = s * bp + c * bq;
                }
                for(int i = 0; i < N; ++i)
                {
                    float vp          = mat4_get(V, i, p);
                    float vq          = mat4_get(V, i, q);
                    mat4_ref(V, i, p) = c * vp - s * vq;
                    mat4_ref(V, i, q) = s * vp + c * vq;
                }
            }
        if(off < EPS * EPS) break;
    }

    for(int j = 0; j < N; ++j)
    {
        float norm = 0.f;
        for(int i = 0; i < N; ++i)
        {
            float b = mat4_get(B, i, j);
            norm += b * b;
        }
        norm     = sqrtf(norm);
        sigma[j] = norm;

        float inv = (norm > EPS) ? 1.f / norm : 0.f;
        for(int i = 0; i < N; ++i)
            mat4_ref(U, i, j) = mat4_get(B, i, j) * inv;
    }

    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            mat4_ref(Vt, i, j) = mat4_get(V, j, i);
}

ATCG_HOST_DEVICE ATCG_INLINE mat6 pseudoinverse(const mat6& A, float rcond = 1e-5f)
{
    constexpr int N = 6;

    mat6 U(0.f), Vt(0.f);
    float sigma[N];

    jacobi_svd6(A, U, sigma, Vt);

    float smax = 0.f;
    for(int i = 0; i < N; ++i)
        smax = fmaxf(smax, sigma[i]);
    float tol = rcond * smax;

    // Pinv[i][j] = sum_k  V[i][k] * sigma_inv[k] * U[j][k]
    //            = sum_k  Vt[k][i] * sigma_inv[k] * U[j][k]
    mat6 Pinv(0.f);
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
        {
            float acc = 0.f;
            for(int k = 0; k < N; ++k)
            {
                float sinv = (sigma[k] > tol) ? 1.f / sigma[k] : 0.f;
                acc += mat6_get(Vt, k, i) * sinv * mat6_get(U, j, k);
            }
            mat6_ref(Pinv, i, j) = acc;
        }

    return Pinv;
}


ATCG_HOST_DEVICE ATCG_INLINE glm::mat4 pseudoinverse(const glm::mat4& A, float rcond = 1e-5f)
{
    constexpr int N = 4;

    glm::mat4 U(0.f), Vt(0.f);
    float sigma[N];

    jacobi_svd4(A, U, sigma, Vt);

    float smax = 0.f;
    for(int i = 0; i < N; ++i)
        smax = fmaxf(smax, sigma[i]);
    float tol = rcond * smax;

    // Pinv[i][j] = sum_k  V[i][k] * sigma_inv[k] * U[j][k]
    //            = sum_k  Vt[k][i] * sigma_inv[k] * U[j][k]
    glm::mat4 Pinv(0.f);
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
        {
            float acc = 0.f;
            for(int k = 0; k < N; ++k)
            {
                float sinv = (sigma[k] > tol) ? 1.f / sigma[k] : 0.f;
                acc += mat4_get(Vt, k, i) * sinv * mat4_get(U, j, k);
            }
            mat4_ref(Pinv, i, j) = acc;
        }

    return Pinv;
}

}    // namespace atcg