#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <BSDF/BSDFVPtrTable.cuh>

struct vec6
{
    ATCG_HOST_DEVICE
    vec6() { a = b = c = d = e = f = 0; }
    union
    {
        glm::vec3 v1;
        float a, b, c;
    };

    union
    {
        glm::vec3 v2;
        float d, e, f;
    };
};

struct mat6
{
    glm::mat3 m00 = glm::mat3(1);
    glm::mat3 m01 = glm::mat3(0);
    glm::mat3 m10 = glm::mat3(0);
    glm::mat3 m11 = glm::mat3(1);
};

ATCG_INLINE ATCG_DEVICE mat6 operator+(const mat6& M, const mat6& N)
{
    mat6 result;

    result.m00 = M.m00 + N.m00;
    result.m01 = M.m01 + N.m01;
    result.m10 = M.m10 + N.m10;
    result.m11 = M.m11 + N.m11;

    return result;
}

ATCG_INLINE ATCG_DEVICE mat6 operator*(const mat6& M, const mat6& N)
{
    mat6 result;

    result.m00 = M.m00 * N.m00 + M.m01 * N.m10;
    result.m01 = M.m00 * N.m01 + M.m01 * N.m11;
    result.m10 = M.m10 * N.m00 + M.m11 * N.m10;
    result.m11 = M.m10 * N.m01 + M.m11 * N.m11;

    return result;
}

struct mat3x6
{
    glm::mat3 m00 = glm::mat3(0);
    glm::mat3 m01 = glm::mat3(0);
};

ATCG_INLINE ATCG_DEVICE mat3x6 operator+(const mat3x6& M, const mat3x6& N)
{
    mat3x6 result;

    result.m00 = M.m00 + N.m00;
    result.m01 = M.m01 + N.m01;

    return result;
}

ATCG_INLINE ATCG_DEVICE mat3x6 operator-(const mat3x6& M, const mat3x6& N)
{
    mat3x6 result;

    result.m00 = M.m00 - N.m00;
    result.m01 = M.m01 - N.m01;

    return result;
}

ATCG_INLINE ATCG_DEVICE mat3x6 operator*(const mat3x6& M, const mat6& N)
{
    mat3x6 result;

    result.m00 = M.m00 * N.m00 + M.m01 * N.m10;
    result.m01 = M.m00 * N.m01 + M.m01 * N.m11;

    return result;
}

ATCG_INLINE ATCG_DEVICE mat3x6 operator*(const glm::mat3& v, const mat3x6& M)
{
    mat3x6 result;

    result.m00 = v * M.m00;
    result.m01 = v * M.m01;

    return result;
}

ATCG_INLINE ATCG_DEVICE vec6 operator*(const glm::vec3& v, const mat3x6& M)
{
    vec6 result;

    result.v1 = v * M.m00;
    result.v2 = v * M.m01;

    return result;
}

ATCG_INLINE ATCG_DEVICE mat6 inverse(const mat6& M)
{
    const auto& A = M.m00;
    const auto& B = M.m01;
    const auto& C = M.m10;
    const auto& D = M.m11;

    auto Ainv = glm::inverse(A);

    auto S    = D - C * Ainv * B;
    auto Sinv = glm::inverse(S);

    mat6 result;

    result.m00 = Ainv + Ainv * B * Sinv * C * Ainv;
    result.m01 = -Ainv * B * Sinv;
    result.m10 = -Sinv * C * Ainv;
    result.m11 = Sinv;

    return result;
}

ATCG_INLINE ATCG_DEVICE glm::mat3 diag(const glm::vec3& v)
{
    glm::mat3 M = glm::mat3(0);

    M[0][0] = v.x;
    M[1][1] = v.y;
    M[2][2] = v.z;

    return M;
}

namespace atcg
{
struct AttachedDiffPathtracingParams
{
    glm::vec3* accumulation_buffer;
    glm::vec3* current_sample;
    glm::vec3* adjoint_y;
    mat3x6* JL_buffer;

    uint32_t image_width;
    uint32_t image_height;

    int32_t* entity_ids;

    OptixTraversableHandle handle;

    TraceParameters surface_trace_params;
    TraceParameters dual_trace_params;
    TraceParameters occlusion_trace_params;

    // Cam data
    float cam_eye[3];
    float U[3];
    float V[3];
    float W[3];
    float fov_y;

    uint32_t frame_counter;
    uint32_t rng_index;

    // Emitter
    uint32_t num_emitters;
    const EmitterVPtrTable** emitters;

    const EmitterVPtrTable* environment_emitter;
};
}    // namespace atcg