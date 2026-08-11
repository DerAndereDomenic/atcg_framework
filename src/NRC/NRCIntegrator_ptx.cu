#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "NRCIntegratorData.cuh"

#include <Core/TraceParameters.h>
#include <DataStructure/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

#include <DataStructure/SampledSpectrum.h>
#include <Integrator/MIS.h>

#include <Neural/Encoding.h>

extern "C"
{
    __constant__ atcg::NRCParams params;
}

ATCG_INLINE ATCG_DEVICE glm::vec3 clampBecauseGLMisTrash(glm::vec3 value, float min, float max)
{
    return glm::vec3(glm::clamp(value.x, min, max), glm::clamp(value.y, min, max), glm::clamp(value.z, min, max));
}

ATCG_INLINE ATCG_DEVICE glm::vec3 mapToBoundingBox(const glm::vec3& pos)
{
    glm::vec3 bbox_min = params.scene_aabb->min;
    glm::vec3 bbox_max = params.scene_aabb->max;

    return clampBecauseGLMisTrash((pos - (bbox_min)) / (bbox_max - bbox_min), 0.0f, 1.0f);
}

extern "C" __global__ void __raygen__sample_generation()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.image_width || launch_idx.y >= params.image_height) return;

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.frame_counter);
    atcg::PCG32 rng(seed);

    atcg::CameraRay camera_ray = params.sensor->generateRay(glm::ivec2(launch_idx.x, launch_idx.y), rng);

    atcg::SampledSpectrum radiance(0);
    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);
    int32_t entity_id                    = -1;

    bool next_ray_valid = true;

    atcg::SurfaceInteraction last_si;
    last_si.pdf = 1.0f;

    for(int n = 0; n < 8; ++n)
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
                bool mis_valid              = last_si.isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf : 0.0f;
                float mis_weight = atcg::BalanceHeuristic::apply(last_si.pdf, emitter_sampling_pdf);
                radiance += mis_weight * camera_ray.importance * si.emitter->evalLight(si, wavelengths);
            }

            // PBR Sampling
            if(si.bsdf)
            {
                // Next-event estimation
                do
                {
                    if(params.num_emitters == 0) break;

                    uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                    float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                    const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                    if(si.emitter == emitter) break;

                    atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, wavelengths, rng);

                    if(emitter_sampling.sampling_pdf == 0) break;

                    emitter_sampling.sampling_pdf *= emitter_selection_pdf;
                    emitter_sampling.radiance_weight_at_receiver /= emitter_selection_pdf;

                    bool occluded = traceOcclusion(params.handle,
                                                   si.position,
                                                   emitter_sampling.direction_to_light,
                                                   1e-3f,
                                                   emitter_sampling.distance_to_light - 1e-3f,
                                                   params.occlusion_trace_params);

                    if(occluded)
                    {
                        break;
                    }

                    atcg::BSDFEvalResult bsdf_result =
                        si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light, wavelengths);

                    float bsdf_pdf   = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0 ||
                                               (int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0
                                           ? 0.0f
                                           : bsdf_result.sample_probability;
                    float mis_weight = atcg::BalanceHeuristic::apply(emitter_sampling.sampling_pdf, bsdf_pdf);

                    radiance += mis_weight * camera_ray.importance * emitter_sampling.radiance_weight_at_receiver *
                                bsdf_result.bsdf_value;
                } while(false);

                auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) == 0)
                    {
                        uint32_t idx = atomicAdd(params.training_samples_queue_index, 1);

                        if(idx < params.max_training_samples)
                        {
                            params.training_samples[idx].position           = si.position;
                            params.training_samples[idx].normal             = si.normal;
                            params.training_samples[idx].outgoing_direction = si.incoming_direction;
                            params.training_samples[idx].weight             = camera_ray.importance;
                            params.training_samples[idx].pixel_index        = pixel_index;
                            params.training_samples[idx].radiance           = radiance;
                        }
                    }

                    camera_ray.ray.origin    = si.position;
                    camera_ray.ray.direction = result.out_dir;
                    camera_ray.importance *= result.bsdf_weight;
                    next_ray_valid = true;

                    last_si     = si;
                    last_si.pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.setInvalid();
                    }
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_si.isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf
                              : 0.0f;
                float mis_weight = last_si.pdf / (last_si.pdf + emitter_sampling_pdf);
                radiance += mis_weight * camera_ray.importance * params.environment_emitter->evalLight(si, wavelengths);
            }
        }
    }

    params.training_sample_radiance[pixel_index] = radiance;
}

extern "C" __global__ void __raygen__train()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.max_training_samples) return;

    atcg::TrainingSample sample    = params.training_samples[launch_idx.x];
    atcg::SampledSpectrum radiance = params.training_sample_radiance[sample.pixel_index];

    glm::vec3 target = (radiance - sample.radiance) / sample.weight;
    half target_x    = __float2half(target.x);
    half target_y    = __float2half(target.y);
    half target_z    = __float2half(target.z);

    OptixCoopVec<half, NRC_INPUT_SIZE> input;

    float encodings[32];
    atcg::sphericalHarmonicEncoding<4>(sample.outgoing_direction.x,
                                       sample.outgoing_direction.y,
                                       sample.outgoing_direction.z,
                                       &encodings[0]);
    atcg::sphericalHarmonicEncoding<4>(sample.normal.x, sample.normal.y, sample.normal.z, &encodings[16]);

    glm::vec3 mapped_position           = mapToBoundingBox(sample.position);
    OptixCoopVec<half, 32> pos_encoding = params.hash_grid->forward(mapped_position);

    for(int i = 0; i < 32; ++i)
    {
        input[i]      = __float2half(encodings[i]);
        input[32 + i] = pos_encoding[i];
    }

    OptixCoopVec<half, NRC_HIDDEN_LAYER_SIZE> hidden[NRC_NUM_HIDDEN_LAYERS + 1];
    OptixCoopVec<half, NRC_HIDDEN_LAYER_SIZE> activations[NRC_NUM_HIDDEN_LAYERS + 1];

    auto output = params.mlp->forward(input, hidden, activations);

    output = atcg::Activation<atcg::ActivationFunction::Sigmoid, NRC_OUTPUT_SIZE>::forward(output);

    // L2 loss
    half loss_scaling = half(1.0f);
    // half loss = loss_scaling *
    //             ((output[0] - target_x) * (output[0] - target_x) + (output[1] - target_y) * (output[1] - target_y) +
    //              (output[2] - target_z) * (output[2] - target_z));

    OptixCoopVec<half, NRC_OUTPUT_SIZE> grad_output(half(0.0f));
    half luminance = half(0.2126f) * output[0] + half(0.7152f) * output[1] + half(0.0722f) * output[2];
    grad_output[0] = half(2.0f) * loss_scaling * (output[0] - target_x) / (luminance * luminance + half(0.01f));
    grad_output[1] = half(2.0f) * loss_scaling * (output[1] - target_y) / (luminance * luminance + half(0.01f));
    grad_output[2] = half(2.0f) * loss_scaling * (output[2] - target_z) / (luminance * luminance + half(0.01f));

    grad_output = atcg::Activation<atcg::ActivationFunction::Sigmoid, NRC_OUTPUT_SIZE>::backward(output, grad_output);

    auto grad_mlp = params.mlp->backward<true>(input, grad_output, hidden, activations);

    OptixCoopVec<half, 32> grad_pos_encoding;
    for(int i = 0; i < 32; ++i)
    {
        grad_pos_encoding[i] = grad_mlp[32 + i];
    }

    params.hash_grid->backward<true>(mapped_position, grad_pos_encoding);
}

extern "C" __global__ void __raygen__render()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.image_width || launch_idx.y >= params.image_height) return;

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.frame_counter);
    atcg::PCG32 rng(seed);

    atcg::CameraRay camera_ray = params.sensor->generateRay(glm::ivec2(launch_idx.x, launch_idx.y), rng);

    atcg::SampledSpectrum radiance(0);
    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);
    int32_t entity_id                    = -1;

    bool next_ray_valid = true;

    atcg::SurfaceInteraction last_si;
    last_si.pdf = 1.0f;

    for(int n = 0; n < 8; ++n)
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
                bool mis_valid              = last_si.isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf : 0.0f;
                float mis_weight = atcg::BalanceHeuristic::apply(last_si.pdf, emitter_sampling_pdf);
                radiance += mis_weight * camera_ray.importance * si.emitter->evalLight(si, wavelengths);
            }

            // PBR Sampling
            if(si.bsdf)
            {
                if((int)(si.bsdf->flags & atcg::BSDFComponentType::AnyDelta) == 0)
                {
                    // Evaluate Radiance cache and terminate
                    OptixCoopVec<half, NRC_INPUT_SIZE> input;

                    float encodings[32];
                    atcg::sphericalHarmonicEncoding<4>(si.incoming_direction.x,
                                                       si.incoming_direction.y,
                                                       si.incoming_direction.z,
                                                       &encodings[0]);
                    atcg::sphericalHarmonicEncoding<4>(si.normal.x, si.normal.y, si.normal.z, &encodings[16]);

                    glm::vec3 mapped_position           = mapToBoundingBox(si.position);
                    OptixCoopVec<half, 32> pos_encoding = params.hash_grid->forward(mapped_position);

                    for(int i = 0; i < 32; ++i)
                    {
                        input[i]      = __float2half(encodings[i]);
                        input[32 + i] = pos_encoding[i];
                    }

                    OptixCoopVec<half, NRC_HIDDEN_LAYER_SIZE> hidden[NRC_NUM_HIDDEN_LAYERS + 1];
                    OptixCoopVec<half, NRC_HIDDEN_LAYER_SIZE> activations[NRC_NUM_HIDDEN_LAYERS + 1];

                    auto output = params.mlp->forward(input, hidden, activations);

                    output = atcg::Activation<atcg::ActivationFunction::Sigmoid, NRC_OUTPUT_SIZE>::forward(output);

                    glm::vec3 cached_radiance =
                        glm::vec3(__half2float(output[0]), __half2float(output[1]), __half2float(output[2]));

                    if(params.visualize_encoding)
                    {
                        cached_radiance =
                            params.encoding_scaling * glm::vec3(__half2float(pos_encoding[params.encoding_channel]),
                                                                -__half2float(pos_encoding[params.encoding_channel]),
                                                                0.0f);
                    }

                    radiance += camera_ray.importance * cached_radiance;

                    break;
                }

                auto result = si.bsdf->sampleBSDF(si, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    camera_ray.ray.origin    = si.position;
                    camera_ray.ray.direction = result.out_dir;
                    camera_ray.importance *= result.bsdf_weight;
                    next_ray_valid = true;

                    last_si     = si;
                    last_si.pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.setInvalid();
                    }
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_si.isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf
                              : 0.0f;
                float mis_weight = last_si.pdf / (last_si.pdf + emitter_sampling_pdf);
                radiance += mis_weight * camera_ray.importance * params.environment_emitter->evalLight(si, wavelengths);
            }
        }
    }

    params.sensor->addSample(glm::ivec3(launch_idx.x, launch_idx.y, 0), radiance, wavelengths);

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