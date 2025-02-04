#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <BSDF/BSDFVPtrTable.cuh>

namespace atcg
{
struct PathtracingParams
{
    glm::u8vec4* output_image;
    glm::vec3* accumulation_buffer;
    uint32_t image_width;
    uint32_t image_height;

    OptixTraversableHandle handle;

    TraceParameters surface_trace_params;
    TraceParameters occlusion_trace_params;

    // Cam data
    float cam_eye[3];
    float U[3];
    float V[3];
    float W[3];
    float fov_y;

    uint32_t frame_counter;

    // Emitter
    uint32_t num_emitters;
    const EmitterVPtrTable** emitters;

    const EmitterVPtrTable* environment_emitter;
};
}