#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <BSDF/BSDFVPtrTable.cuh>
#include <Sensor/SensorVPtrTable.cuh>
#include <DataStructure/BoundingBox.h>

#include <Neural/DeviceMLP.cuh>
#include <Neural/DeviceHashGrid.h>

#define NRC_INPUT_SIZE                   64
#define NRC_OUTPUT_SIZE                  8
#define NRC_HIDDEN_LAYER_SIZE            64
#define NRC_NUM_HIDDEN_LAYERS            5
#define NRC_HASH_GRID_LEVELS             16
#define NRC_HASH_GRID_FEATURES_PER_LEVEL 2
#define NRC_NUM_WEIGHTS                                                                                                \
    ((NRC_INPUT_SIZE * NRC_HIDDEN_LAYER_SIZE) +                                                                        \
     (NRC_HIDDEN_LAYER_SIZE * NRC_HIDDEN_LAYER_SIZE) * (NRC_NUM_HIDDEN_LAYERS) +                                       \
     (NRC_HIDDEN_LAYER_SIZE * NRC_OUTPUT_SIZE))
#define NRC_NUM_BIASES (NRC_HIDDEN_LAYER_SIZE * (NRC_NUM_HIDDEN_LAYERS + 1) + NRC_OUTPUT_SIZE)

namespace atcg
{

using NRCDeviceMLP      = DeviceMLP<NRC_NUM_HIDDEN_LAYERS, NRC_INPUT_SIZE, NRC_HIDDEN_LAYER_SIZE, NRC_OUTPUT_SIZE>;
using NRCDeviceHashGrid = DeviceHashGrid<half, NRC_HASH_GRID_LEVELS, NRC_HASH_GRID_FEATURES_PER_LEVEL>;

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
    atcg::BoundingBox* scene_aabb;

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

    NRCDeviceMLP* mlp;
    NRCDeviceHashGrid* hash_grid;
};
}    // namespace atcg