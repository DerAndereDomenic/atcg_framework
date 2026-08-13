#pragma once

#include <Core/Optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <Material/BSDFVPtrTable.h>
#include <Material/MediumVPtrTable.h>
#include <Sensor/SensorVPtrTable.cuh>

namespace atcg
{
struct VolPathtracingParams
{
    uint32_t image_width;
    uint32_t image_height;

    int32_t* entity_ids;

    OptixTraversableHandle handle;

    TraceParameters surface_trace_params;
    TraceParameters occlusion_trace_params;

    uint32_t frame_counter;

    // Emitter
    uint32_t num_emitters;
    const EmitterVPtrTable** emitters;

    const EmitterVPtrTable* environment_emitter;

    const SensorVPtrTable* sensor;
};
}