#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "AttachedDiffPathtracingData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>
#include <Math/Functions.h>
#include <DataStructure/Frame.h>
#include <Integrator/MIS.h>
#include <Utils/HostDevice.h>

#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

extern "C"
{
    __constant__ atcg::AttachedDiffPathtracingParams params;
}

struct RayContext
{
    bool valid;
    atcg::AnyInteraction si0;
    atcg::AnyInteraction si1;
    CuDiff::Dual<6, glm::vec3> last_normal;
    CuDiff::Dual<6, glm::vec2> last_uv;

    glm::vec3 throughput;
    glm::vec3 radiance;

    glm::vec3 delta_y;
    atcg::mat6x3 JL;

    atcg::MediumVPtrTable* current_medium = nullptr;
};

extern "C" __global__ void __raygen__forward()
{
    uint3 launch_idx = optixGetLaunchIndex();

    if(launch_idx.x >= params.image_width || launch_idx.y >= params.image_height) return;

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.rng_index);
    atcg::PCG32 rng(seed);

    atcg::SampledWavelengths wavelengths = atcg::SampledWavelengths::sampleSpectrum(rng.nextFloat(), 380.0f, 780.0f);

    glm::vec2 jitter = rng.next2d();
    float u_         = (((float)launch_idx.x + jitter.x) / (float)params.image_width - 0.5f) * 2.0f;
    float v_         = (((float)launch_idx.y + jitter.y) / (float)params.image_height - 0.5f) * 2.0f;

    glm::vec3 cam_eye = glm::make_vec3(params.cam_eye);
    glm::vec3 U       = glm::make_vec3(params.U) * (float)params.image_width / (float)params.image_height;
    glm::vec3 V       = glm::make_vec3(params.V);
    glm::vec3 W       = glm::make_vec3(params.W) / glm::tan(glm::radians(params.fov_y / 2.0f));

    RayContext ray;

    glm::vec3 ray_origin    = cam_eye;
    glm::vec3 ray_direction = glm::normalize((u_ * U + v_ * V + W));
    ray.radiance =
        atcg::select(params.diff_mode == atcg::DiffMode::FORWARD, glm::vec3(0), params.current_sample[pixel_index]);
    ray.throughput = glm::vec3(1);
    ray.valid      = false;
    ray.JL =
        atcg::select(params.diff_mode == atcg::DiffMode::FORWARD, atcg::mat6x3(0.0f), params.JL_buffer[pixel_index]);
    ray.delta_y =
        atcg::select(params.diff_mode == atcg::DiffMode::FORWARD, glm::vec3(0), params.adjoint_y[pixel_index]);

    atcg::SurfaceInteraction si0;
    si0.position           = ray_origin;
    si0.reference_frame    = atcg::Frame<glm::vec3>(ray_direction);
    si0.incoming_direction = ray_direction;

    atcg::mat6x3 Jb = atcg::mat6x3(0);
    atcg::mat6 Jray = atcg::mat6(1);

    atcg::DualSurfaceInteraction dsi1;
    dsi1.incoming_position  = CuDiff::Dual<6, glm::vec3>(ray_origin);
    dsi1.incoming_direction = CuDiff::Dual<6, glm::vec3>(ray_direction);
    atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                             ray_origin,
                                                             ray_direction,
                                                             0.001f,
                                                             1e16f,
                                                             &dsi1,
                                                             params.dual_trace_params);
    auto si1        = dsi1.toSi();
    ray.last_normal = dsi1.normal;
    ray.last_uv     = dsi1.uv;


    if(si1.isValid())
    {
        ray.valid = true;
    }

    ray.si0      = si0;
    ray.si0->pdf = 1.0f;
    ray.si1      = si1;

    atcg::mat4x6 frame_ray_0 = atcg::mat4x6(glm::mat2x3(si0.reference_frame.localX(), si0.reference_frame.localY()),
                                            glm::mat2x3(0.0f),
                                            glm::mat2x3(0.0f),
                                            glm::mat2x3(si1.reference_frame.localX(), si1.reference_frame.localY()));

    for(int n = 0; n < 8; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        atcg::AnyInteraction si0     = ray.si0;
        atcg::SurfaceInteraction si1 = ray.si1;

        auto [x0, x1] = CuDiff::make_variables<6>(si0->position, si1.position);
        auto distance = CuDiff::length(x1 - x0);
        auto w        = (x1 - x0) / CuDiff::max(distance, 1e-5f);

        atcg::DualSurfaceInteraction dsi;
        dsi.position           = x1;
        dsi.incoming_direction = w;
        dsi.incoming_distance  = distance;
        dsi.normal             = ray.last_normal;
        dsi.uv                 = ray.last_uv;

        // glm::mat2x3 frame0 = glm::mat2x3(si0->reference_frame.localX(), si0->reference_frame.localY());
        glm::mat2x3 frame1 = glm::mat2x3(si1.reference_frame.localX(), si1.reference_frame.localY());

        // si is valid by contruction if(si.valid)
        {
            // Check for light source
            glm::vec3 Le(0.0f);
            atcg::mat6x3 JLe = atcg::mat6x3(0.0f);
            if(si1.emitter)
            {
                bool mis_valid              = si0->isValid();
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    atcg::select(mis_valid, si1.emitter->evalLightSamplingPdf(si0, si1) * emitter_selection_pdf, 0.0f);
                float mis_weight  = atcg::PowerHeuristic<1>::apply(si0->pdf, emitter_sampling_pdf);
                auto light_result = si1.emitter->evalLightForward(dsi, wavelengths);
                Le                = mis_weight * light_result.radiance_weight_at_receiver;

                if(params.diff_mode == atcg::DiffMode::FORWARD)
                {
                    ray.radiance += ray.throughput * Le;
                }
                else
                {
                    ray.radiance -= ray.throughput * Le;
                }

                JLe = light_result.dLe_dx0x1;

                JLe = JLe * Jray;
            }

            if(params.diff_mode == atcg::DiffMode::FORWARD)
            {
                ray.JL += atcg::diag(ray.throughput) * JLe + atcg::diag(Le) * Jb;
            }

            // PBR Sampling
            if(si1.bsdf)
            {
                // Next-event estimation
                do
                {
                    if(params.num_emitters == 0) break;

                    uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                    float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                    const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                    if(si1.emitter == emitter) break;

                    atcg::EmitterDualSamplingResult emitter_sampling =
                        emitter->sampleLightForward(dsi, wavelengths, rng);

                    if(emitter_sampling.sampling_pdf == 0) break;

                    emitter_sampling.sampling_pdf *= emitter_selection_pdf;
                    emitter_sampling.radiance_weight_at_receiver =
                        emitter_sampling.radiance_weight_at_receiver / emitter_selection_pdf;

                    bool occluded = traceOcclusion(params.handle,
                                                   si1.position,
                                                   emitter_sampling.direction_to_light,
                                                   1e-3f,
                                                   emitter_sampling.distance_to_light - 1e-3f,
                                                   params.occlusion_trace_params);

                    if(occluded)
                    {
                        break;
                    }

                    atcg::BSDFDualEvalResult bsdf_result =
                        si1.bsdf->evalBSDFForward(dsi, emitter_sampling.direction_to_light, wavelengths);

                    float bsdf_pdf = atcg::select((int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0 ||
                                                      (int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0,
                                                  0.0f,
                                                  bsdf_result.sample_probability);
                    float mis_weight = atcg::PowerHeuristic<1>::apply(emitter_sampling.sampling_pdf, bsdf_pdf);

                    glm::vec3 throughput_nee = ray.throughput * bsdf_result.bsdf_value;

                    glm::vec3 radiance_nee = mis_weight * throughput_nee * emitter_sampling.radiance_weight_at_receiver;

                    if(params.diff_mode == atcg::DiffMode::FORWARD)
                    {
                        ray.radiance += radiance_nee;
                    }
                    else
                    {
                        ray.radiance -= radiance_nee;
                    }

                    auto JLe_nee = emitter_sampling.dLe_dx0x1 / emitter_selection_pdf;

                    JLe_nee = JLe_nee * Jray;

                    auto Jbsdf_nee = bsdf_result.dbsdf_dx0x1;

                    Jbsdf_nee = Jbsdf_nee * Jray;

                    if(params.diff_mode == atcg::DiffMode::FORWARD)
                    {
                        atcg::mat6x3 Jb_nee =
                            atcg::diag(bsdf_result.bsdf_value) * Jb + atcg::diag(ray.throughput) * Jbsdf_nee;

                        ray.JL += mis_weight * (atcg::diag(emitter_sampling.radiance_weight_at_receiver) * Jb_nee +
                                                atcg::diag(throughput_nee) * JLe_nee);
                    }
                    else
                    {
                        ray.JL -= (atcg::diag(radiance_nee / bsdf_result.bsdf_value) * Jbsdf_nee +
                                   mis_weight * atcg::diag(ray.throughput * bsdf_result.bsdf_value) * JLe_nee);

                        glm::vec3 grad_out =
                            (ray.delta_y * (radiance_nee + 1e-4f)) / (glm::vec3(bsdf_result.bsdf_value) + 1e-4f);
                        si1.bsdf->evalBSDFBackward(si1, emitter_sampling.direction_to_light.val(), grad_out);
                    }


                } while(false);

                auto rng_copy = rng;
                auto result   = si1.bsdf->sampleBSDFForward(dsi, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    atcg::DualSurfaceInteraction next_dsi;
                    next_dsi.incoming_position  = x1;
                    next_dsi.incoming_direction = result.out_dir;

                    atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                                             si1.position,
                                                                             result.out_dir.val(),
                                                                             0.001f,
                                                                             1e16f,
                                                                             &next_dsi,
                                                                             params.dual_trace_params);

                    if(!next_dsi.isValid())
                    {
                        ray.valid = false;
                        continue;
                    }

                    glm::mat2x3 frame2 =
                        glm::mat2x3(next_dsi.reference_frame.localX(), next_dsi.reference_frame.localY());

                    auto Jray_ = next_dsi.dx1x2_dx0x1;

                    auto Jbsdf = result.dbsdf_dx0x1;


                    Jbsdf = Jbsdf * Jray;
                    Jray  = Jray_ * Jray;
                    Jray += atcg::mat6(0.01f * glm::sign(rng.nextFloat() - 0.5f));    // Regularization

                    if(params.diff_mode == atcg::DiffMode::FORWARD)
                    {
                        Jb = atcg::diag(result.bsdf_weight) * Jb + atcg::diag(ray.throughput) * Jbsdf;
                    }
                    else
                    {
                        ray.JL -=
                            (atcg::diag(ray.radiance / result.bsdf_weight) * Jbsdf + atcg::diag(ray.throughput) * JLe);


                        atcg::mat4x6 frame_ray_n = atcg::mat4x6(frame1, glm::mat2x3(0.0f), glm::mat2x3(0.0f), frame2);

                        auto J_ray_uv = atcg::transpose(frame_ray_n) * (Jray * frame_ray_0);

                        auto JL = ray.JL * frame_ray_0;

                        auto Jrayinv            = glm::inverse(J_ray_uv);
                        glm::mat4x3 JL_         = JL * Jrayinv;    // dL/d(du1v1, du2v2)
                        glm::mat3x2 du2v2dwo    = glm::transpose(frame2) * next_dsi.dxdw;
                        glm::mat3x4 du1v1u2v2dw = glm::mat3x4(glm::vec4(glm::vec2(0), du2v2dwo[0]),
                                                              glm::vec4(glm::vec2(0), du2v2dwo[1]),
                                                              glm::vec4(glm::vec2(0), du2v2dwo[2]));

                        // 𝛿𝜋 += backward_grad(bsdf_value, 𝛿𝐿 ∗ 𝐿 / bsdf_value)
                        // = 1/pi * dL * L / (albedo / pi) = dL * L / albedo
                        glm::vec3 dLdbsdf = (ray.delta_y * (ray.radiance + 1e-4f)) / (result.bsdf_weight + 1e-4f);
                        glm::vec3 dLdwo   = ray.delta_y * (JL_ * du1v1u2v2dw);

                        si1.bsdf->sampleBSDFBackward(si1, rng_copy, dLdbsdf, dLdwo);
                    }

                    si1.pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        si1.setInvalid();
                    }

                    ray.si0         = si1;
                    ray.si1         = next_dsi.toSi();
                    ray.last_normal = next_dsi.normal;
                    ray.last_uv     = next_dsi.uv;

                    ray.throughput *= result.bsdf_weight;
                    ray.valid = next_dsi.isValid();
                }
            }
        }
        // TODO
        // else
        // {
        //     if(params.environment_emitter)
        //     {
        //         bool mis_valid              = last_si.valid;
        //         float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
        //         float emitter_sampling_pdf =
        //             mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf
        //                       : 0.0f;
        //         float mis_weight = 1.0f;    // last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
        //         ray.radiance += mis_weight * ray.throughput * params.environment_emitter->evalLight(si);
        //     }
        // }
    }

    if(params.diff_mode == atcg::DiffMode::FORWARD)
    {
        params.current_sample[pixel_index] = ray.radiance;
        params.JL_buffer[pixel_index]      = ray.JL;
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

extern "C" __global__ void __miss__dual()
{
    atcg::DualSurfaceInteraction* si = getPayloadDataPointer<atcg::DualSurfaceInteraction>();
    float3 optix_world_dir           = optixGetWorldRayDirection();

    si->setInvalid();
}

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}