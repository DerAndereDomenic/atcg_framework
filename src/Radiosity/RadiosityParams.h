#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Shape/MeshShape.h>

struct RadiosityParams
{
    OptixTraversableHandle handle;
    atcg::TraceParameters occlusion_trace_params;

    uint32_t n_faces;
    float* form_factors;
    atcg::MeshShapeData* shape;
};