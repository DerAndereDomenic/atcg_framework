#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <BSDF/BSDFVPtrTable.cuh>
#include <Medium/MediumVPtrTable.cuh>
#include <Integrator/DiffMode.h>
#include <Sensor/SensorVPtrTable.cuh>

namespace atcg
{
struct VolDiffPathtracingParams
{
    glm::vec3* current_sample;
    glm::vec3* adjoint_y;

    uint32_t image_width;
    uint32_t image_height;

    OptixTraversableHandle handle;

    TraceParameters surface_trace_params;
    TraceParameters occlusion_trace_params;

    const SensorVPtrTable* sensor;

    uint32_t rng_index;

    // Emitter
    uint32_t num_emitters;
    const EmitterVPtrTable** emitters;

    const EmitterVPtrTable* environment_emitter;

    DiffMode diff_mode;
};
}    // namespace atcg