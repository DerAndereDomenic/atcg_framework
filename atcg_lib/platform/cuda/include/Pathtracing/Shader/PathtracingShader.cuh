#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Pathtracing/BSDF/BSDFModels.cuh>
#include <Pathtracing/Emitter/EmitterModels.cuh>

struct Params
{
    glm::vec3* accumulation_buffer;
    glm::u8vec4* output_image;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t frame_counter;

    OptixTraversableHandle handle;

    // Emitters
    const atcg::EmitterVPtrTable** emitters;
    uint32_t num_emitters;

    // Cam data
    float cam_eye[3];
    float U[3];
    float V[3];
    float W[3];
    float fov_y;

    // Skybox
    const atcg::EmitterVPtrTable* environment_emitter;
};
