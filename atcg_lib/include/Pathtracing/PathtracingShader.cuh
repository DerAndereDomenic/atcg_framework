#pragma once

#include <Core/glm.h>
#include <Pathtracing/EmitterModels.cuh>
#include <Pathtracing/BSDFModels.cuh>
#include <Pathtracing/AccelerationStructure.h>

struct PathtracingParams
{
    glm::vec3* accumulation_buffer;
    glm::u8vec4* output_image;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t frame_counter;

    // OptixTraversableHandle handle;
    atcg::IASAccelerationStructure* handle;

    // TraceParameters surface_trace_params;
    // TraceParameters occlusion_trace_params;

    // Emitters
    const atcg::EmitterVPtrTable** emitters;
    uint32_t num_emitters;

    const atcg::BSDFVPtrTable** bsdfs;
    uint32_t num_bsdfs;

    // Cam data
    float cam_eye[3];
    float U[3];
    float V[3];
    float W[3];
    float fov_y;

    // Skybox
    const atcg::EmitterVPtrTable* environment_emitter;
};