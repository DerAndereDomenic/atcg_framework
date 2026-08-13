#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Integrator/PhotonMapData.cuh>

#include <Core/TraceParameters.h>
#include <DataStructure/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

#include <DataStructure/SampledSpectrum.h>
#include <Integrator/MIS.h>
#include <BSDF/BSDFFunctions.h>

#include <cuBQL/traversal/fixedRadiusQuery.h>

extern "C"
{
    __constant__ atcg::PhotonMapParams params;
}

ATCG_DEVICE ATCG_INLINE atcg::PhotonSamplingResult samplePhoton(const atcg::SampledWavelengths& wavelengths,
                                                                atcg::PCG32& rng)
{
    uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

    const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

    auto result = emitter->samplePhoton(wavelengths, rng);

    result.pdf /= static_cast<float>(params.num_emitters);
    result.radiance_weight *= static_cast<float>(params.num_emitters);

    return result;
}

extern "C" __global__ void __raygen__sample_photons()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.photons_per_launch) return;

    uint32_t seed = atcg::sampleTEA64(launch_idx.x, params.frame_counter);
    atcg::PCG32 rng(seed);

    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);

    atcg::PhotonSamplingResult photon = samplePhoton(wavelengths, rng);

    bool next_ray_valid = true;

    glm::vec3 origin                 = photon.position;
    glm::vec3 direction              = photon.direction;
    atcg::SampledSpectrum throughput = photon.radiance_weight;

    float initial_weight = throughput.maxComponent();

    for(int n = 0; n < PHOTON_MAP_TRACE_DEPTH; ++n)
    {
        if(!next_ray_valid) break;
        next_ray_valid = false;

        float q = throughput.maxComponent() / initial_weight;

        if(rng.nextFloat() > q)
        {
            break;
        }
        throughput /= q;


        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             origin,
                                                             direction,
                                                             0.001f,
                                                             1e16f,
                                                             &si,
                                                             params.surface_trace_params);

        if(!si.isValid())
        {
            break;
        }


        // PBR Sampling
        if(!si.bsdf)
        {
            break;
        }

        if(!atcg::hasMaterialFlag(si.bsdf->flags, atcg::MaterialFlag::AnyDelta))
        {
            int photon_index = atomicAdd(params.photon_index, 1);

            if(photon_index >= params.max_num_photons)
            {
                break;
            }

            atcg::PhotonMapData photon_data;
            photon_data.position   = si.position;
            photon_data.direction  = -direction;
            photon_data.normal     = atcg::faceForward(si.normal, -direction);
            photon_data.throughput = throughput;

            params.photon_data[photon_index] = photon_data;
            params.photon_bounds[photon_index] =
                cuBQL::box3f().including(cuBQL::vec3f(si.position.x, si.position.y, si.position.z));
        }

        auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);

        if(result.sample_probability > 0.0f)
        {
            origin    = si.position;
            direction = result.out_dir;
            throughput *= result.bsdf_weight;
            next_ray_valid = true;
        }
    }
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.image_width || launch_idx.y >= params.image_height) return;

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.frame_counter);
    atcg::PCG32 rng(seed);

    atcg::CameraRay camera_ray = params.sensor->generateRay(glm::ivec2(launch_idx.x, launch_idx.y), rng);

    if(!camera_ray.valid)
    {
        return;
    }

    if(params.frame_counter == 0)
    {
        // Reset PPM
        params.photon_gather_data[pixel_index].photon_count     = 0;
        params.photon_gather_data[pixel_index].gather_radius_sq = PHOTON_MAP_GATHER_RADIUS_SQ;
        params.photon_gather_data[pixel_index].gathered_power   = atcg::SampledSpectrum(0.0f);
    }

    atcg::SampledSpectrum mc_radiance(0);
    atcg::SampledSpectrum ppm_radiance(0);
    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);
    int32_t entity_id                    = -1;

    bool next_ray_valid = true;

    for(int n = 0; n < PHOTON_MAP_TRACE_DEPTH; ++n)
    {
        if(!next_ray_valid) break;
        next_ray_valid = false;

        atcg::SurfaceInteraction si;
        atcg::traceWithDataPointer<atcg::SurfaceInteraction>(params.handle,
                                                             camera_ray.ray.origin,
                                                             camera_ray.ray.direction,
                                                             0.001f,
                                                             1e16f,
                                                             &si,
                                                             params.surface_trace_params);

        if(si.isValid() && n == 0)
        {
            entity_id = si.entity_id;
        }

        if(si.isValid())
        {
            // Check for light source
            if(si.emitter)
            {
                mc_radiance += camera_ray.importance * si.emitter->evalLight(si, wavelengths);
            }

            // PBR Sampling
            if(si.bsdf)
            {
                auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);
                if(!atcg::hasMaterialFlag(si.bsdf->flags, atcg::MaterialFlag::AnyDelta) &&
                   !atcg::hasMaterialFlag(result.flags, atcg::MaterialFlag::AnyDelta))
                {
                    cuBQL::vec3f query_pos(si.position.x, si.position.y, si.position.z);

                    float radius_sq = params.photon_gather_data[pixel_index].gather_radius_sq;
                    int num_photons = 0;
                    atcg::SampledSpectrum photon_power(0.0f);
                    atcg::PhotonMapData* photon_data = params.photon_data;
                    auto accumulator =
                        [&si, photon_data, &num_photons, &photon_power, &wavelengths](const uint32_t primID)
                    {
                        auto current_photon = photon_data[primID];
                        glm::vec3 normal    = atcg::faceForward(current_photon.normal, current_photon.direction);
                        if(glm::dot(normal, current_photon.normal) > 0.2f)
                        {
                            auto bsdf_val = si.bsdf->evalBSDF(si, current_photon.direction, wavelengths);
                            float NdotL   = glm::max(1e-3f, glm::dot(normal, current_photon.direction));
                            // Divide by NdotL because this cancels out with the bsdf's cosine when substituting
                            // radiance with power
                            photon_power += bsdf_val.bsdf_value * current_photon.throughput / NdotL;

                            ++num_photons;
                        }

                        return CUBQL_CONTINUE_TRAVERSAL;
                    };

                    cuBQL::fixedRadiusQuery::forEachPrim(accumulator, *params.photon_bvh, query_pos, radius_sq);

                    // PPM
                    atcg::SampledSpectrum gathered_total_power = params.photon_gather_data[pixel_index].gathered_power;
                    gathered_total_power += camera_ray.importance * photon_power;
                    float N = (float)params.photon_gather_data[pixel_index].photon_count;
                    float M = (float)num_photons;

                    float gather_photon_count = N + PHOTON_MAP_REDUCTION_FACTOR * M;

                    if(M != 0)
                    {
                        float reduction_factor_sq = (float)gather_photon_count / (float)(N + M);
                        radius_sq *= reduction_factor_sq;
                        gathered_total_power *= reduction_factor_sq;
                    }

                    params.photon_gather_data[pixel_index].gather_radius_sq = radius_sq;
                    params.photon_gather_data[pixel_index].gathered_power   = gathered_total_power;
                    params.photon_gather_data[pixel_index].photon_count     = gather_photon_count;

                    break;
                }


                if(result.sample_probability > 0.0f)
                {
                    camera_ray.ray.origin    = si.position;
                    camera_ray.ray.direction = result.out_dir;
                    camera_ray.importance *= result.bsdf_weight;
                    next_ray_valid = true;
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                mc_radiance += camera_ray.importance * params.environment_emitter->evalLight(si, wavelengths);
            }
        }
    }

    if(params.frame_counter > 0)
    {
        // Mix with previous subframes if present!
        const float a                                    = 1.0f / static_cast<float>(params.frame_counter + 1);
        const atcg::SampledSpectrum prev_output_radiance = params.photon_gather_data[pixel_index].direct_radiance;
        mc_radiance                                      = (1.0f - a) * prev_output_radiance + a * mc_radiance;
    }

    atcg::SampledSpectrum gathered_total_power = params.photon_gather_data[pixel_index].gathered_power;
    float radius_sq                            = params.photon_gather_data[pixel_index].gather_radius_sq;

    int total_emitted_photons = (params.frame_counter + 1) * PHOTON_MAP_PHOTONS_PER_LAUNCH;
    ppm_radiance              = gathered_total_power / ((float)total_emitted_photons * glm::pi<float>() * radius_sq);

    params.photon_gather_data[pixel_index].direct_radiance = mc_radiance;
    params.sensor->addSample(glm::ivec3(launch_idx.x, launch_idx.y, 0),
                             mc_radiance + ppm_radiance,
                             wavelengths);    // Set sample count to 0 so no accumulation happens in film

    if(params.entity_ids)
    {
        params.entity_ids[pixel_index] = entity_id;
    }
}

extern "C" __global__ void __miss__ms()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();
    float3 optix_world_dir       = optixGetWorldRayDirection();
    glm::vec3 ray_dir            = glm::make_vec3((float*)&optix_world_dir);

    si->setInvalid();
    si->incoming_direction = ray_dir;
}

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}