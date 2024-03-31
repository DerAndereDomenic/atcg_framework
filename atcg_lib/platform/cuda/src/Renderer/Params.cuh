#pragma once

#include <optix.h>
#include <Core/glm.h>

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
    bool hasSkybox;
    glm::vec3* skybox_data;
    uint32_t skybox_width;
    uint32_t skybox_height;
};

// These structs represent the data blocks of our SBT records
struct RayGenData
{
    // No data needed
};
struct HitGroupData
{
    glm::vec3* positions;
    glm::vec3* normals;
    glm::vec3* uvs;
    glm::u32vec3* faces;
};

struct MissData
{
    // No data needed
};

// SBT record with an appropriately aligned and sized data block template
template<typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;