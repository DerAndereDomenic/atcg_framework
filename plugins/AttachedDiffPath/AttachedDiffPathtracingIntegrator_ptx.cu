#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "AttachedDiffPathtracingData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>
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

    atcg::vec6 Jpdf;

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

    auto camera_ray = params.sensor->generateRay(glm::ivec2(launch_idx.x, launch_idx.y), rng);

    RayContext ray;

    glm::vec3 ray_origin    = camera_ray.ray.origin;
    glm::vec3 ray_direction = glm::normalize(camera_ray.ray.direction);
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

        atcg::DualSurfaceInteraction dsi0;
        dsi0.position = x0;

        atcg::DualSurfaceInteraction dsi;
        dsi.position           = x1;
        dsi.incoming_direction = w;
        dsi.incoming_distance  = distance;
        dsi.normal             = ray.last_normal;
        dsi.uv                 = ray.last_uv;

        // glm::mat2x3 frame0 = glm::mat2x3(si0->reference_frame.localX(), si0->reference_frame.localY());
        glm::mat2x3 frame1 = glm::mat2x3(si1.reference_frame.localX(), si1.reference_frame.localY());

        // Check for light source
        glm::vec3 Le(0.0f);
        atcg::mat6x3 JLe = atcg::mat6x3(0.0f);
        float mis_weight = 1.0f;
        atcg::vec6 Jmis  = atcg::vec6(0.0f);
        if(si1.emitter)
        {
            bool mis_valid              = si0->isValid();
            float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
            auto emitter_sampling_pdf =
                atcg::select(mis_valid,
                             si1.emitter->evalLightSamplingPdfForward(dsi0, dsi) * emitter_selection_pdf,
                             CuDiff::Dual<6, float>(0.0f));

            atcg::vec6 Jpdf_light = atcg::vec6(glm::vec3(emitter_sampling_pdf.derivative(0),
                                                         emitter_sampling_pdf.derivative(1),
                                                         emitter_sampling_pdf.derivative(2)),
                                               glm::vec3(emitter_sampling_pdf.derivative(3),
                                                         emitter_sampling_pdf.derivative(4),
                                                         emitter_sampling_pdf.derivative(5)));

            Jpdf_light           = Jpdf_light * Jray;
            atcg::vec6 Jpdf_bsdf = ray.Jpdf;

            float mis_denom   = emitter_sampling_pdf.val() + si0->pdf;
            mis_weight        = si0->pdf / mis_denom;
            auto light_result = si1.emitter->evalLightForward(dsi, wavelengths);
            Le                = light_result.radiance_weight_at_receiver;

            if(params.diff_mode == atcg::DiffMode::FORWARD)
            {
                ray.radiance += mis_weight * ray.throughput * Le;
            }
            else
            {
                ray.radiance -= mis_weight * ray.throughput * Le;
            }

            float dmis_dpbsdf  = emitter_sampling_pdf.val() / (mis_denom * mis_denom);
            float dmis_dplight = -si0->pdf / (mis_denom * mis_denom);

            Jmis = dmis_dpbsdf * Jpdf_bsdf + dmis_dplight * Jpdf_light;


            JLe = light_result.dLe_dx0x1;

            JLe = JLe * Jray;

            if(params.diff_mode == atcg::DiffMode::FORWARD)
            {
                ray.JL += atcg::diag(mis_weight * ray.throughput) * JLe + atcg::diag(mis_weight * Le) * Jb +
                          atcg::mat6x3(glm::outerProduct(ray.throughput * Le, Jmis.a),
                                       glm::outerProduct(ray.throughput * Le, Jmis.b));
            }
        }


        // PBR Sampling
        if(!si1.bsdf)
        {
            break;
        }

        // Next-event estimation
        do
        {
            if(params.num_emitters == 0) break;

            uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

            float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

            const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

            if(si1.emitter == emitter) break;

            atcg::EmitterDualSamplingResult emitter_sampling = emitter->sampleLightForward(dsi, wavelengths, rng);

            if(emitter_sampling.sampling_pdf == 0) break;

            emitter_sampling.sampling_pdf *= emitter_selection_pdf;
            emitter_sampling.radiance_weight_at_receiver =
                emitter_sampling.radiance_weight_at_receiver / emitter_selection_pdf;
            emitter_sampling.dpdf_dx0x1 = emitter_sampling.dpdf_dx0x1 / emitter_selection_pdf;
            emitter_sampling.dLe_dx0x1  = emitter_sampling.dLe_dx0x1 / emitter_selection_pdf;

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

            float mis_denom  = emitter_sampling.sampling_pdf + bsdf_pdf;
            float mis_weight = emitter_sampling.sampling_pdf / mis_denom;

            glm::vec3 Le_nee = emitter_sampling.radiance_weight_at_receiver;

            glm::vec3 throughput_nee = ray.throughput * bsdf_result.bsdf_value;

            glm::vec3 radiance_nee = throughput_nee * Le_nee;

            if(params.diff_mode == atcg::DiffMode::FORWARD)
            {
                ray.radiance += mis_weight * radiance_nee;
            }
            else
            {
                ray.radiance -= mis_weight * radiance_nee;
            }

            atcg::vec6 Jpdf_light = emitter_sampling.dpdf_dx0x1;
            atcg::vec6 Jpdf_bsdf  = atcg::select((int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0 ||
                                                     (int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0,
                                                 atcg::vec6(0.0f),
                                                 bsdf_result.dpdf_dx0x1);

            Jpdf_light = Jpdf_light * Jray;
            Jpdf_bsdf  = Jpdf_bsdf * Jray;

            float dmis_dpbsdf  = -emitter_sampling.sampling_pdf / (mis_denom * mis_denom);
            float dmis_dplight = bsdf_pdf / (mis_denom * mis_denom);

            atcg::vec6 Jmis = dmis_dpbsdf * Jpdf_bsdf + dmis_dplight * Jpdf_light;

            auto JLe_nee   = emitter_sampling.dLe_dx0x1 * Jray;
            auto Jbsdf_nee = bsdf_result.dbsdf_dx0x1 * Jray;

            if(params.diff_mode == atcg::DiffMode::FORWARD)
            {
                atcg::mat6x3 Jb_nee = atcg::diag(bsdf_result.bsdf_value) * Jb + atcg::diag(ray.throughput) * Jbsdf_nee;

                ray.JL += atcg::diag(mis_weight * Le_nee) * Jb_nee + atcg::diag(mis_weight * throughput_nee) * JLe_nee +
                          atcg::mat6x3(glm::outerProduct(throughput_nee * Le_nee, Jmis.a),
                                       glm::outerProduct(throughput_nee * Le_nee, Jmis.b));
            }
            else
            {
                ray.JL -=
                    (atcg::diag(mis_weight * (radiance_nee + 1e-4f) / (bsdf_result.bsdf_value + 1e-4f)) * Jbsdf_nee +
                     atcg::diag(mis_weight * throughput_nee) * JLe_nee +
                     atcg::mat6x3(glm::outerProduct(throughput_nee * Le_nee, Jmis.a),
                                  glm::outerProduct(throughput_nee * Le_nee, Jmis.b)));

                glm::vec3 dLdbsdf = (ray.delta_y * mis_weight * ray.throughput * Le_nee);

                glm::vec3 dLdp = ray.delta_y * (dmis_dpbsdf * throughput_nee * Le_nee);

                auto gradients =
                    si1.bsdf->evalBSDFBackward(si1, emitter_sampling.direction_to_light.val(), dLdbsdf, dLdp);

                for(int i = 0; i < gradients.num_payloads; ++i)
                {
                    if(i >= params.num_aovs) break;
                    params.aov_buffers[i][pixel_index] += gradients.payload[i];
                }
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

            glm::mat2x3 frame2 = glm::mat2x3(next_dsi.reference_frame.localX(), next_dsi.reference_frame.localY());

            auto Jray_ = next_dsi.dx1x2_dx0x1;

            auto Jbsdf = result.dbsdf_dx0x1;
            auto Jpdf  = result.dpdf_dx0x1;


            Jbsdf = Jbsdf * Jray;
            Jpdf  = Jpdf * Jray;
            Jray  = Jray_ * Jray;
            Jray += atcg::mat6(0.01f * glm::sign(rng.nextFloat() - 0.5f));    // Regularization

            if(params.diff_mode == atcg::DiffMode::FORWARD)
            {
                Jb = atcg::diag(result.bsdf_weight) * Jb + atcg::diag(ray.throughput) * Jbsdf;
            }
            else
            {
                ray.JL -= (atcg::diag((ray.radiance + 1e-4f) / (result.bsdf_weight + 1e-4f)) * Jbsdf +
                           atcg::diag(mis_weight * ray.throughput) * JLe +
                           atcg::mat6x3(glm::outerProduct(ray.throughput * Le, Jmis.a),
                                        glm::outerProduct(ray.throughput * Le, Jmis.b)));

                bool mis_valid        = (int)(result.flags & atcg::BSDFComponentType::AnyDelta) == 0;
                float mis_weight_next = atcg::select(mis_valid, 1.0f, 0.0f);
                float dmis_dpbsdf     = 0.0f;

                if(next_dsi.emitter)
                {
                    float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                    auto emitter_sampling_pdf   = atcg::select(
                        mis_valid,
                        next_dsi.emitter->evalLightSamplingPdf(si1, next_dsi.toSi()) * emitter_selection_pdf,
                        0.0f);

                    float mis_denom_next = emitter_sampling_pdf + result.sample_probability;
                    mis_weight_next      = result.sample_probability / mis_denom_next;
                    dmis_dpbsdf          = emitter_sampling_pdf / (mis_denom_next * mis_denom_next);
                }


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
                glm::vec3 dLdpdf  = (ray.delta_y * (ray.radiance + 1e-4f)) / (mis_weight_next + 1e-4f) * dmis_dpbsdf;
                glm::vec3 dLdwo   = ray.delta_y * (JL_ * du1v1u2v2dw);

                auto gradients = si1.bsdf->sampleBSDFBackward(si1, rng_copy, dLdbsdf, dLdpdf, dLdwo);
                for(int i = 0; i < gradients.num_payloads; ++i)
                {
                    if(i >= params.num_aovs) break;
                    params.aov_buffers[i][pixel_index] += gradients.payload[i];
                }
            }

            si1.pdf  = result.sample_probability;
            ray.Jpdf = Jpdf;

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

    // if(dsi1.isValid())
    // {
    //     glm::mat3 dx0dx0 = glm::mat3(1.0f);
    //     glm::mat3 dx1dx0 = glm::mat3(1.0f) - glm::outerProduct(ray_direction, dsi1.normal.val()) /
    //                                              glm::dot(ray_direction, dsi1.normal.val());

    //     auto J_total = ray.JL.m00 * dx0dx0 + ray.JL.m01 * dx1dx0;
    //     ray.JL.m00   = J_total;
    // }

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