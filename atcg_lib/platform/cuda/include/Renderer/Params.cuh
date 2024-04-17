#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Renderer/BSDFModels.cuh>
#include <Renderer/Emitter.cuh>

struct Params
{
    glm::vec3* accumulation_buffer;
    glm::u8vec4* output_image;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t frame_counter;

    OptixTraversableHandle handle;

    // Cam data
    float cam_eye[3];
    float U[3];
    float V[3];
    float W[3];

    // Skybox
    const atcg::EmitterVPtrTable* environment_emitter;
};

struct HitGroupData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* uvs;
    glm::u32vec3* faces;

    const atcg::BSDFVPtrTable* bsdf;
    const atcg::EmitterVPtrTable* emitter;
};
