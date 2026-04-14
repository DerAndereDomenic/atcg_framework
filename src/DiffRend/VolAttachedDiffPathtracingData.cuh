#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <BSDF/BSDFVPtrTable.cuh>
#include <Math/mat6.h>
#include "DiffMode.h"

namespace atcg
{

struct VolAttachedDiffPathtracingParams
{
    DiffMode diff_mode;

    glm::vec3* current_sample;
    glm::vec3* adjoint_y;
    atcg::mat6x3* JL_buffer;

    uint32_t image_width;
    uint32_t image_height;

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

    uint32_t rng_index;

    // Emitter
    uint32_t num_emitters;
    const EmitterVPtrTable** emitters;

    const EmitterVPtrTable* environment_emitter;

    bool debug;
};
}    // namespace atcg