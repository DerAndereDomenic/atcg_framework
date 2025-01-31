#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>

namespace atcg
{
struct PathtracingParams
{
    glm::u8vec4* output_image;
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
};
}