#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <BSDF/BSDFVPtrTable.cuh>
#include <Sensor/SensorVPtrTable.cuh>

#include <Neural/DeviceMLP.cuh>
#include <Neural/DeviceHashGrid.h>

namespace atcg
{

struct TrainingSample
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 outgoing_direction;

    SampledSpectrum weight;
    SampledSpectrum radiance;
    int pixel_index;
};

struct NRCParams
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

    TrainingSample* training_samples;
    int* training_samples_queue_index;
    uint32_t max_training_samples;
    SampledSpectrum* training_sample_radiance;

    DeviceMLP<3, 64, 64, 8>* mlp;
    DeviceHashGrid<half, 16, 2>* hash_grid;
};
}