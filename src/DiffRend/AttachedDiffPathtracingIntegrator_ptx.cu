#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "AttachedDiffPathtracingData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>
#include <Math/Functions.h>

#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

extern "C"
{
    __constant__ atcg::AttachedDiffPathtracingParams params;
}

struct RayContext
{
    bool valid;
    atcg::SurfaceInteraction si0;
    atcg::SurfaceInteraction si1;
    CuDiff::Dual<6, glm::vec3> last_normal;
    CuDiff::Dual<6, glm::vec2> last_uv;

    glm::vec3 throughput;
    glm::vec3 radiance;

    glm::vec3 delta_y;
    glm::mat4x3 JL;
};

extern "C" __global__ void __raygen__forward()
{
    uint3 launch_idx = optixGetLaunchIndex();

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
    ray.radiance            = glm::vec3(0);
    ray.throughput          = glm::vec3(1);
    ray.valid               = false;
    ray.JL                  = glm::mat4x3(0);
    int32_t entity_id       = -1;

    atcg::SurfaceInteraction si0;
    si0.valid              = true;
    si0.position           = ray_origin;
    si0.normal             = ray_direction;
    si0.incoming_direction = ray_direction;

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

    glm::mat4x3 Jb = glm::mat4x3(0);
    glm::mat4 Jray = glm::mat4(1);

    atcg::DualSurfaceInteraction dsi1;
    dsi1.incoming_position  = ray_origin;
    dsi1.incoming_direction = ray_direction;
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

    if(si1.valid)
    {
        entity_id = si1.entity_id;
        ray.valid = true;
    }

    ray.si0 = si0;
    ray.si1 = si1;

    for(int n = 0; n < 8; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        auto [x0, x1] = CuDiff::make_variables<6>(ray.si0.position, ray.si1.position);
        auto distance = CuDiff::length(x1 - x0);
        auto w        = (x1 - x0) / CuDiff::max(distance, 1e-5f);

        atcg::DualSurfaceInteraction dsi;
        dsi.valid              = true;
        dsi.position           = x1;
        dsi.incoming_direction = w;
        dsi.incoming_distance  = distance;
        dsi.normal             = ray.last_normal;
        dsi.uv                 = ray.last_uv;

        auto frame0_ = atcg::Math::compute_local_frame(ray.si0.normal);
        auto frame1_ = atcg::Math::compute_local_frame(ray.si1.normal);

        glm::mat2x3 frame0 = glm::mat2x3(frame0_[0], frame0_[1]);
        glm::mat2x3 frame1 = glm::mat2x3(frame1_[0], frame1_[1]);

        // si is valid by contruction if(si.valid)
        {
            // Check for light source
            CuDiff::Dual<6, glm::vec3> Le;
            glm::mat4x3 JLe = glm::mat4x3(0);
            if(ray.si1.emitter)
            {
                // bool mis_valid = last_si.valid;
                // float emitter_sampling_pdf = mis_valid ? ray.si1.emitter->evalLightSamplingPdf(last_si, ray.si1) :
                // 0.0f; float mis_weight = last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                Le = ray.si1.emitter->evalLightDual(dsi, wavelengths);
                ray.radiance += ray.throughput * Le.val();

                glm::mat3 JLe_dx0 = glm::mat3(Le.derivative(0), Le.derivative(1), Le.derivative(2));
                glm::mat3 JLe_dx1 = glm::mat3(Le.derivative(3), Le.derivative(4), Le.derivative(5));

                glm::mat2x3 JLe_du0v0 = JLe_dx0 * frame0;
                glm::mat2x3 JLe_du1v1 = JLe_dx1 * frame1;

                JLe = glm::mat4x3(glm::vec3(JLe_du0v0[0]),
                                  glm::vec3(JLe_du0v0[1]),
                                  glm::vec3(JLe_du1v1[0]),
                                  glm::vec3(JLe_du1v1[1]));

                JLe = JLe * Jray;
            }

            ray.JL += diag(ray.throughput) * JLe + diag(Le.val()) * Jb;

            // PBR Sampling
            if(ray.si1.bsdf)
            {
                // Next-event estimation
                // do
                // {
                //     if(params.num_emitters == 0) break;

                //     uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                //     float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                //     const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                //     if(si.emitter == emitter) break;

                //     atcg::EmitterSamplingResult emitter_sampling =
                //         emitter->sampleLight(si, rng);    // TODO: Differentiate

                //     if(emitter_sampling.sampling_pdf == 0) break;

                //     emitter_sampling.sampling_pdf *= emitter_selection_pdf;

                //     bool occluded = traceOcclusion(params.handle,
                //                                    si.position,
                //                                    emitter_sampling.direction_to_light,
                //                                    1e-3f,
                //                                    emitter_sampling.distance_to_light - 1e-3f,
                //                                    params.occlusion_trace_params);

                //     if(occluded)
                //     {
                //         break;
                //     }

                //     atcg::BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light);

                //     float bsdf_pdf   = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0
                //                            ? 0.0f
                //                            : bsdf_result.sample_probability;
                //     float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + bsdf_pdf);

                //     glm::vec3 radiance_nee = mis_weight * ray.throughput *
                //                              emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value *
                //                              glm::abs(glm::dot(si.normal, emitter_sampling.direction_to_light));

                //     ray.radiance += radiance_nee;
                // } while(false);

                auto result = ray.si1.bsdf->sampleBSDFForward(dsi, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    atcg::DualSurfaceInteraction next_dsi;
                    next_dsi.incoming_position  = dsi.position;
                    next_dsi.incoming_direction = result.out_dir;

                    atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                                             ray.si1.position,
                                                                             result.out_dir.val(),
                                                                             0.001f,
                                                                             1e16f,
                                                                             &next_dsi,
                                                                             params.dual_trace_params);

                    if(!next_dsi.valid)
                    {
                        ray.valid = false;
                        continue;
                    }

                    auto frame2_ = atcg::Math::compute_local_frame(next_dsi.normal.val());

                    glm::mat2x3 frame2 = glm::mat2x3(frame2_[0], frame2_[1]);

                    glm::mat3 dx1_dx0 = glm::mat3(0);
                    glm::mat3 dx2_dx0 = glm::mat3(next_dsi.position.derivative(0),
                                                  next_dsi.position.derivative(1),
                                                  next_dsi.position.derivative(2));
                    glm::mat3 dx1_dx1 = glm::mat3(1);
                    glm::mat3 dx2_dx1 = glm::mat3(next_dsi.position.derivative(3),
                                                  next_dsi.position.derivative(4),
                                                  next_dsi.position.derivative(5));

                    glm::mat2 du1v1_du0v0 = glm::transpose(frame1) * dx1_dx0 * frame0;
                    glm::mat2 du2v2_du0v0 = glm::transpose(frame2) * dx2_dx0 * frame0;
                    glm::mat2 du1v1_du1v1 = glm::transpose(frame1) * dx1_dx1 * frame1;
                    glm::mat2 du2v2_du1v1 = glm::transpose(frame2) * dx2_dx1 * frame1;

                    // Construct 4x4 Jacobian ((du1v1_du0v0, du1v1_du1v1), (du2v2_du0v0, du2v2_du1v1))
                    glm::mat4 Jray_ = glm::mat4(glm::vec4(du1v1_du0v0[0], du2v2_du0v0[0]),
                                                glm::vec4(du1v1_du0v0[1], du2v2_du0v0[1]),
                                                glm::vec4(du1v1_du1v1[0], du2v2_du1v1[0]),
                                                glm::vec4(du1v1_du1v1[1], du2v2_du1v1[1]));

                    glm::mat3 Jbsdf_dx0 = glm::mat3(result.bsdf_weight.derivative(0),
                                                    result.bsdf_weight.derivative(1),
                                                    result.bsdf_weight.derivative(2));
                    glm::mat3 Jbsdf_dx1 = glm::mat3(result.bsdf_weight.derivative(3),
                                                    result.bsdf_weight.derivative(4),
                                                    result.bsdf_weight.derivative(5));

                    glm::mat2x3 Jbsdf_du0v0 = Jbsdf_dx0 * frame0;
                    glm::mat2x3 Jbsdf_du1v1 = Jbsdf_dx1 * frame1;

                    glm::mat4x3 Jbsdf = glm::mat4x3(glm::vec3(Jbsdf_du0v0[0]),
                                                    glm::vec3(Jbsdf_du0v0[1]),
                                                    glm::vec3(Jbsdf_du1v1[0]),
                                                    glm::vec3(Jbsdf_du1v1[1]));

                    Jbsdf = Jbsdf * Jray;
                    Jray  = Jray_ * Jray;
                    Jray += 0.01f * glm::mat4(1) * glm::sign(rng.nextFloat() - 0.5f);    // Regularization

                    Jb = diag(result.bsdf_weight.val()) * Jb + diag(ray.throughput) * Jbsdf;

                    // if(params.debug)
                    // {
                    //     printf("%f\n", glm::determinant(Jray_));
                    // }

                    ray.si0         = ray.si1;
                    ray.si1         = next_dsi.toSi();
                    ray.last_normal = next_dsi.normal;
                    ray.last_uv     = next_dsi.uv;

                    ray.throughput *= result.bsdf_weight.val();
                    ray.valid = next_dsi.valid;

                    // last_si       = si;
                    // last_bsdf_pdf = result.sample_probability;

                    // if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    // {
                    //     last_si.valid = false;
                    // }
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

    params.current_sample[pixel_index] = ray.radiance;
    params.JL_buffer[pixel_index]      = ray.JL;

    if(params.frame_counter > 0)
    {
        // Mix with previous subframes if present!
        const float a                        = 1.0f / static_cast<float>(params.frame_counter + 1);
        const glm::vec3 prev_output_radiance = params.accumulation_buffer[pixel_index];
        ray.radiance                         = glm::lerp(prev_output_radiance, ray.radiance, a);
    }

    params.accumulation_buffer[pixel_index] = ray.radiance;

    if(params.entity_ids)
    {
        params.entity_ids[pixel_index] = entity_id;
    }
}

extern "C" __global__ void __raygen__backward()
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
    ray.radiance            = params.accumulation_buffer[pixel_index];
    ray.throughput          = glm::vec3(1);
    ray.valid               = false;
    ray.JL                  = params.JL_buffer[pixel_index];
    ray.delta_y             = params.adjoint_y[pixel_index];
    int32_t entity_id       = -1;

    atcg::SurfaceInteraction si0;
    si0.valid              = true;
    si0.position           = ray_origin;
    si0.normal             = ray_direction;
    si0.incoming_direction = ray_direction;

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

    glm::mat4 Jray = glm::mat4(1);

    atcg::DualSurfaceInteraction dsi1;
    dsi1.incoming_position  = ray_origin;
    dsi1.incoming_direction = ray_direction;
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

    if(si1.valid)
    {
        entity_id = si1.entity_id;
        ray.valid = true;
    }

    ray.si0 = si0;
    ray.si1 = si1;

    for(int n = 0; n < 8; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        auto [x0, x1] = CuDiff::make_variables<6>(ray.si0.position, ray.si1.position);
        auto distance = CuDiff::length(x1 - x0);
        auto w        = (x1 - x0) / CuDiff::max(distance, 1e-5f);

        atcg::DualSurfaceInteraction dsi;
        dsi.valid              = true;
        dsi.position           = x1;
        dsi.incoming_direction = w;
        dsi.incoming_distance  = distance;
        dsi.normal             = ray.last_normal;
        dsi.uv                 = ray.last_uv;

        auto frame0_ = atcg::Math::compute_local_frame(ray.si0.normal);
        auto frame1_ = atcg::Math::compute_local_frame(ray.si1.normal);

        glm::mat2x3 frame0 = glm::mat2x3(frame0_[0], frame0_[1]);
        glm::mat2x3 frame1 = glm::mat2x3(frame1_[0], frame1_[1]);


        {
            // Check for light source
            CuDiff::Dual<6, glm::vec3> Le;
            glm::mat4x3 JLe = glm::mat4x3(0);
            if(ray.si1.emitter)
            {
                // bool mis_valid = last_si.valid;
                // float emitter_sampling_pdf = mis_valid ? ray.si1.emitter->evalLightSamplingPdf(last_si, ray.si1) :
                // 0.0f; float mis_weight = last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                Le = ray.si1.emitter->evalLightDual(dsi, wavelengths);
                ray.radiance -= ray.throughput * Le.val();

                glm::mat3 JLe_dx0 = glm::mat3(Le.derivative(0), Le.derivative(1), Le.derivative(2));
                glm::mat3 JLe_dx1 = glm::mat3(Le.derivative(3), Le.derivative(4), Le.derivative(5));

                glm::mat2x3 JLe_du0v0 = JLe_dx0 * frame0;
                glm::mat2x3 JLe_du1v1 = JLe_dx1 * frame1;

                JLe = glm::mat4x3(glm::vec3(JLe_du0v0[0]),
                                  glm::vec3(JLe_du0v0[1]),
                                  glm::vec3(JLe_du1v1[0]),
                                  glm::vec3(JLe_du1v1[1]));

                JLe = JLe * Jray;
            }

            // PBR Sampling
            if(ray.si1.bsdf)
            {
                // Next-event estimation
                // do
                // {
                //     if(params.num_emitters == 0) break;

                //     uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

                //     float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

                //     const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

                //     if(si.emitter == emitter) break;

                //     atcg::EmitterSamplingResult emitter_sampling = emitter->sampleLight(si, rng);

                //     if(emitter_sampling.sampling_pdf == 0) break;

                //     emitter_sampling.sampling_pdf *= emitter_selection_pdf;

                //     bool occluded = traceOcclusion(params.handle,
                //                                    si.position,
                //                                    emitter_sampling.direction_to_light,
                //                                    1e-3f,
                //                                    emitter_sampling.distance_to_light - 1e-3f,
                //                                    params.occlusion_trace_params);

                //     if(occluded)
                //     {
                //         break;
                //     }

                //     atcg::BSDFEvalResult bsdf_result = si.bsdf->evalBSDF(si, emitter_sampling.direction_to_light);

                //     float bsdf_pdf   = (int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0
                //                            ? 0.0f
                //                            : bsdf_result.sample_probability;
                //     float mis_weight = emitter_sampling.sampling_pdf / (emitter_sampling.sampling_pdf + bsdf_pdf);

                //     glm::vec3 radiance_nee = mis_weight * ray.throughput *
                //                              emitter_sampling.radiance_weight_at_receiver * bsdf_result.bsdf_value *
                //                              glm::abs(glm::dot(si.normal, emitter_sampling.direction_to_light));

                //     glm::vec3 grad_out = (ray.delta_y * (radiance_nee + 1e-4f)) / (bsdf_result.bsdf_value + 1e-4f);
                //     si.bsdf->evalBSDFBackward(si, emitter_sampling.direction_to_light, grad_out);

                //     ray.radiance -= radiance_nee;
                // } while(false);

                auto rng_copy = rng;
                auto result   = ray.si1.bsdf->sampleBSDFForward(dsi, wavelengths, rng);

                if(result.sample_probability > 0.0f)
                {
                    atcg::DualSurfaceInteraction next_dsi;
                    next_dsi.incoming_position  = dsi.position;
                    next_dsi.incoming_direction = result.out_dir;

                    atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                                             ray.si1.position,
                                                                             result.out_dir.val(),
                                                                             0.001f,
                                                                             1e16f,
                                                                             &next_dsi,
                                                                             params.dual_trace_params);

                    if(!next_dsi.valid)
                    {
                        ray.valid = false;
                        continue;
                    }

                    auto frame2_ = atcg::Math::compute_local_frame(next_dsi.normal.val());

                    glm::mat2x3 frame2 = glm::mat2x3(frame2_[0], frame2_[1]);

                    glm::mat3 dx1_dx0 = glm::mat3(0);
                    glm::mat3 dx2_dx0 = glm::mat3(next_dsi.position.derivative(0),
                                                  next_dsi.position.derivative(1),
                                                  next_dsi.position.derivative(2));
                    glm::mat3 dx1_dx1 = glm::mat3(1);
                    glm::mat3 dx2_dx1 = glm::mat3(next_dsi.position.derivative(3),
                                                  next_dsi.position.derivative(4),
                                                  next_dsi.position.derivative(5));

                    glm::mat2 du1v1_du0v0 = glm::transpose(frame1) * dx1_dx0 * frame0;
                    glm::mat2 du2v2_du0v0 = glm::transpose(frame2) * dx2_dx0 * frame0;
                    glm::mat2 du1v1_du1v1 = glm::transpose(frame1) * dx1_dx1 * frame1;
                    glm::mat2 du2v2_du1v1 = glm::transpose(frame2) * dx2_dx1 * frame1;

                    // Construct 4x4 Jacobian ((du1v1_du0v0, du1v1_du1v1), (du2v2_du0v0, du2v2_du1v1))
                    glm::mat4 Jray_ = glm::mat4(glm::vec4(du1v1_du0v0[0], du2v2_du0v0[0]),
                                                glm::vec4(du1v1_du0v0[1], du2v2_du0v0[1]),
                                                glm::vec4(du1v1_du1v1[0], du2v2_du1v1[0]),
                                                glm::vec4(du1v1_du1v1[1], du2v2_du1v1[1]));

                    glm::mat3 Jbsdf_dx0 = glm::mat3(result.bsdf_weight.derivative(0),
                                                    result.bsdf_weight.derivative(1),
                                                    result.bsdf_weight.derivative(2));
                    glm::mat3 Jbsdf_dx1 = glm::mat3(result.bsdf_weight.derivative(3),
                                                    result.bsdf_weight.derivative(4),
                                                    result.bsdf_weight.derivative(5));

                    glm::mat2x3 Jbsdf_du0v0 = Jbsdf_dx0 * frame0;
                    glm::mat2x3 Jbsdf_du1v1 = Jbsdf_dx1 * frame1;

                    glm::mat4x3 Jbsdf = glm::mat4x3(glm::vec3(Jbsdf_du0v0[0]),
                                                    glm::vec3(Jbsdf_du0v0[1]),
                                                    glm::vec3(Jbsdf_du1v1[0]),
                                                    glm::vec3(Jbsdf_du1v1[1]));

                    Jbsdf = Jbsdf * Jray;
                    Jray  = Jray_ * Jray;
                    Jray += 0.01f * glm::mat4(1) * glm::sign(rng.nextFloat() - 0.5f);    // Regularization

                    ray.JL -= (diag(ray.radiance / result.bsdf_weight.val()) * Jbsdf + diag(ray.throughput) * JLe);

                    auto Jrayinv    = glm::inverse(Jray);
                    glm::mat4x3 JL_ = ray.JL * Jrayinv;

                    // 𝛿𝜋 += backward_grad(bsdf_value, 𝛿𝐿 ∗ 𝐿 / bsdf_value)
                    // = 1/pi * dL * L / (albedo / pi) = dL * L / albedo
                    glm::vec3 dLdbsdf = (ray.delta_y * (ray.radiance + 1e-4f)) / (result.bsdf_weight.val() + 1e-4f);
                    glm::vec2 dLdwo   = glm::zw(ray.delta_y * JL_);    // Only v2?

                    ray.si1.bsdf->sampleBSDFBackward(ray.si1, rng_copy, dLdbsdf, dLdwo);

                    ray.si0         = ray.si1;
                    ray.si1         = next_dsi.toSi();
                    ray.last_normal = next_dsi.normal;
                    ray.last_uv     = next_dsi.uv;

                    ray.throughput *= result.bsdf_weight.val();
                    ray.valid = next_dsi.valid;

                    // last_si       = si;
                    // last_bsdf_pdf = result.sample_probability;

                    // if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    // {
                    //     last_si.valid = false;
                    // }
                }
            }
        }
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
        //         ray.radiance -= mis_weight * ray.throughput * params.environment_emitter->evalLight(si);
        //     }
        // }
    }
}

extern "C" __global__ void __miss__ms()
{
    atcg::SurfaceInteraction* si = getPayloadDataPointer<atcg::SurfaceInteraction>();
    float3 optix_world_dir       = optixGetWorldRayDirection();
    glm::vec3 ray_dir            = glm::make_vec3((float*)&optix_world_dir);

    si->valid              = false;
    si->incoming_distance  = std::numeric_limits<float>::infinity();
    si->incoming_direction = ray_dir;
}

extern "C" __global__ void __miss__dual()
{
    atcg::DualSurfaceInteraction* si = getPayloadDataPointer<atcg::DualSurfaceInteraction>();
    float3 optix_world_dir           = optixGetWorldRayDirection();
    glm::vec3 ray_dir                = glm::make_vec3((float*)&optix_world_dir);

    si->valid              = false;
    si->incoming_distance  = std::numeric_limits<float>::infinity();
    si->incoming_direction = ray_dir;
}

extern "C" __global__ void __miss__occlusion()
{
    setOcclusionPayload(false);
}