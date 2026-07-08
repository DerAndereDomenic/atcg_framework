#pragma once

#include <optix.h>
#include <Core/glm.h>
#include <Core/TraceParameters.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <BSDF/BSDFVPtrTable.cuh>
#include <Sensor/SensorVPtrTable.cuh>
#include <cuBQL/bvh.h>
#include <cuBQL/math/box.h>

#define PHOTON_MAP_MAX_NUM_PHOTONS    1000000
#define PHOTON_MAP_TRACE_DEPTH        8
#define PHOTON_MAP_PHOTONS_PER_LAUNCH ((PHOTON_MAP_MAX_NUM_PHOTONS) / (PHOTON_MAP_TRACE_DEPTH))
#define PHOTON_MAP_GATHER_RADIUS      0.25f
#define PHOTON_MAP_GATHER_RADIUS_SQ   ((PHOTON_MAP_GATHER_RADIUS) * (PHOTON_MAP_GATHER_RADIUS))
#define PHOTON_MAP_REDUCTION_FACTOR   0.7f

namespace atcg
{

struct PhotonMapData
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 normal;
    SampledSpectrum throughput;
};

struct PhotonGatherData
{
    uint32_t photon_count           = 0;
    float gather_radius_sq          = PHOTON_MAP_GATHER_RADIUS_SQ;
    SampledSpectrum gathered_power  = SampledSpectrum(0.0f);
    SampledSpectrum direct_radiance = SampledSpectrum(0.0f);
};

struct PhotonMapParams
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

    // Photon data
    uint32_t photons_per_launch;
    uint32_t max_num_photons;

    PhotonMapData* photon_data;
    cuBQL::box3f* photon_bounds;
    cuBQL::bvh3f* photon_bvh;
    int* photon_index;

    PhotonGatherData* photon_gather_data;
};
}    // namespace atcg