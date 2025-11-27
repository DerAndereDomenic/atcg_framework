#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "AttachedDiffPathtracingData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>

#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

extern "C"
{
    __constant__ atcg::AttachedDiffPathtracingParams params;
}

struct RayContext
{
    bool valid;
    CuDiff::Dual<4, glm::vec3> origin;
    CuDiff::Dual<4, glm::vec3> direction;
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
    float theta_            = glm::acos(glm::clamp(ray_direction.y, -1.0f, 1.0f));
    float phi_              = glm::atan2(ray_direction.z, ray_direction.x);

    auto [u, v, phi, theta] = CuDiff::make_variables<4>(u_, v_, phi_, theta_);

    auto sinTheta = CuDiff::sin(theta);

    auto x = sinTheta * CuDiff::cos(phi);
    auto y = CuDiff::cos(theta);
    auto z = sinTheta * CuDiff::sin(phi);

    CuDiff::Dual<4, glm::vec3> dir = CuDiff::wrap(x, y, z);

    ray.origin        = cam_eye + 0.01f * CuDiff::normalize((u * U + v * V + W));
    ray.direction     = dir;
    ray.radiance      = glm::vec3(0);
    ray.throughput    = glm::vec3(1);
    ray.valid         = true;
    ray.JL            = glm::mat4x3(0);
    int32_t entity_id = -1;

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

    glm::mat4x3 Jb = glm::mat4x3(0);
    glm::mat4 Jray = glm::mat4(1);

    for(int n = 0; n < 8; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        atcg::DualSurfaceInteraction dsi;
        dsi.incoming_position  = ray.origin;
        dsi.incoming_direction = ray.direction;
        atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                                 ray.origin,
                                                                 ray.direction,
                                                                 0.001f,
                                                                 1e16f,
                                                                 &dsi,
                                                                 params.dual_trace_params);
        atcg::SurfaceInteraction si = dsi.toSi();

        if(si.valid && n == 0)
        {
            entity_id = si.entity_id;
        }

        if(si.valid)
        {
            // Check for light source
            CuDiff::Dual<4, glm::vec3> Le;
            glm::mat4x3 JLe;
            if(si.emitter)
            {
                bool mis_valid             = last_si.valid;
                float emitter_sampling_pdf = mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) : 0.0f;
                float mis_weight           = 1.0f;    // last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                Le                         = si.emitter->evalLightDual(dsi);
                ray.radiance += mis_weight * ray.throughput * Le.val();

                JLe = glm::mat4x3(Le.derivative(0), Le.derivative(1), Le.derivative(2), Le.derivative(3));

                JLe = JLe * Jray;
            }

            ray.JL = ray.JL + diag(ray.throughput) * JLe + diag(Le.val()) * Jb;

            // PBR Sampling
            if(si.bsdf)
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

                //     ray.radiance += radiance_nee;
                // } while(false);

                auto result = si.bsdf->sampleBSDFForward(dsi, rng);

                if(result.sample_probability > 0.0f)
                {
                    auto [dx, dy, dz] = CuDiff::unwrap(result.out_dir);

                    auto theta_n = CuDiff::acos(CuDiff::clamp(dy, -1.0f, 1.0f));
                    auto phi_n   = CuDiff::atan2(dz, dx);

                    // Jray
                    float duip1dui    = dsi.u_surface.derivative(0);
                    float duip1p1dvi  = dsi.u_surface.derivative(1);
                    float duip1p1dphi = dsi.u_surface.derivative(2);
                    float duip1dtheta = dsi.u_surface.derivative(3);
                    float dvip1dui    = dsi.v_surface.derivative(0);
                    float dvip1dvi    = dsi.v_surface.derivative(1);
                    float dvip1dphi   = dsi.v_surface.derivative(2);
                    float dvip1dtheta = dsi.v_surface.derivative(3);

                    float dthetaip1dui    = theta_n.derivative(0);
                    float dthetaip1p1dvi  = theta_n.derivative(1);
                    float dthetaip1p1dphi = theta_n.derivative(2);
                    float dthetaip1dtheta = theta_n.derivative(3);
                    float dphiip1dui      = phi_n.derivative(0);
                    float dphiip1dvi      = phi_n.derivative(1);
                    float dphiip1dphi     = phi_n.derivative(2);
                    float dphiip1dtheta   = phi_n.derivative(3);

                    glm::mat4 Jray_ = glm::mat4(glm::vec4(duip1dui, dvip1dui, dphiip1dui, dthetaip1dui),
                                                glm::vec4(duip1p1dvi, dvip1dvi, dphiip1dvi, dthetaip1p1dvi),
                                                glm::vec4(duip1p1dphi, dvip1dphi, dphiip1dphi, dthetaip1p1dphi),
                                                glm::vec4(duip1dtheta, dvip1dtheta, dphiip1dtheta, dthetaip1dtheta));

                    // -------------------

                    // Jbsdf

                    glm::mat4x3 Jbsdf = glm::mat4x3(result.bsdf_weight.derivative(0),
                                                    result.bsdf_weight.derivative(1),
                                                    result.bsdf_weight.derivative(2),
                                                    result.bsdf_weight.derivative(3));

                    Jbsdf = Jbsdf * Jray;

                    // -------------------

                    Jb = diag(result.bsdf_weight.val()) * Jb + diag(ray.throughput) * Jbsdf;

                    Jray = Jray_ * Jray;

                    thrust::tie(u, v, phi, theta) =
                        CuDiff::make_variables<4>(dsi.u_surface.val(), dsi.v_surface.val(), phi_n.val(), theta_n.val());

                    auto sinTheta = CuDiff::sin(theta);

                    auto x = sinTheta * CuDiff::cos(phi);
                    auto y = CuDiff::cos(theta);
                    auto z = sinTheta * CuDiff::sin(phi);

                    ray.origin    = (1.0f - u - v) * dsi.P0 + u * dsi.P1 + v * dsi.P2;
                    ray.direction = CuDiff::wrap(x, y, z);
                    ray.throughput *= result.bsdf_weight.val();
                    ray.valid = true;

                    last_si       = si;
                    last_bsdf_pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.valid = false;
                    }
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_si.valid;
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf
                              : 0.0f;
                float mis_weight = 1.0f;    // last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                ray.radiance += mis_weight * ray.throughput * params.environment_emitter->evalLight(si);
            }
        }
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

    uint32_t pixel_index = launch_idx.x + params.image_width * launch_idx.y;
    uint64_t seed        = atcg::sampleTEA64(pixel_index, params.rng_index);
    atcg::PCG32 rng(seed);

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
    float theta_            = glm::acos(glm::clamp(ray_direction.y, -1.0f, 1.0f));
    float phi_              = glm::atan2(ray_direction.z, ray_direction.x);

    auto [u, v, phi, theta] = CuDiff::make_variables<4>(u_, v_, phi_, theta_);

    auto sinTheta = CuDiff::sin(theta);

    auto x = sinTheta * CuDiff::cos(phi);
    auto y = CuDiff::cos(theta);
    auto z = sinTheta * CuDiff::sin(phi);

    CuDiff::Dual<4, glm::vec3> dir = CuDiff::wrap(x, y, z);

    ray.origin        = cam_eye + 0.01f * CuDiff::normalize((u * U + v * V + W));
    ray.direction     = dir;
    ray.radiance      = params.accumulation_buffer[pixel_index];
    ray.delta_y       = params.adjoint_y[pixel_index];
    ray.JL            = params.JL_buffer[pixel_index];
    ray.throughput    = glm::vec3(1);
    ray.valid         = true;
    int32_t entity_id = -1;

    atcg::SurfaceInteraction last_si;
    float last_bsdf_pdf = 1.0f;

    glm::mat4 Jray = glm::mat4(1);

    for(int n = 0; n < 8; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        atcg::DualSurfaceInteraction dsi;
        dsi.incoming_position  = ray.origin;
        dsi.incoming_direction = ray.direction;
        atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                                 ray.origin,
                                                                 ray.direction,
                                                                 0.001f,
                                                                 1e16f,
                                                                 &dsi,
                                                                 params.dual_trace_params);
        atcg::SurfaceInteraction si = dsi.toSi();

        if(si.valid && n == 0)
        {
            entity_id = si.entity_id;
        }

        if(si.valid)
        {
            // TODO: No NEE for now
            // Check for light source
            CuDiff::Dual<4, glm::vec3> Le;
            glm::mat4x3 JLe;
            if(si.emitter)
            {
                bool mis_valid             = last_si.valid;
                float emitter_sampling_pdf = mis_valid ? si.emitter->evalLightSamplingPdf(last_si, si) : 0.0f;
                float mis_weight           = 1.0f;    // last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                Le                         = si.emitter->evalLightDual(dsi);
                ray.radiance -= mis_weight * ray.throughput * Le.val();

                JLe = glm::mat4x3(Le.derivative(0), Le.derivative(1), Le.derivative(2), Le.derivative(3));

                JLe = JLe * Jray;
            }

            // PBR Sampling
            if(si.bsdf)
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
                //     si.bsdf->backwardGrad(si, emitter_sampling.direction_to_light, grad_out);

                //     ray.radiance -= radiance_nee;
                // } while(false);

                auto result = si.bsdf->sampleBSDFForward(dsi, rng);

                if(result.sample_probability > 0.0f)
                {
                    auto [dx, dy, dz] = CuDiff::unwrap(result.out_dir);

                    auto theta_n = CuDiff::acos(CuDiff::clamp(dy, -1.0f, 1.0f));
                    auto phi_n   = CuDiff::atan2(dz, dx);

                    // Jray
                    float duip1dui    = dsi.u_surface.derivative(0);
                    float duip1p1dvi  = dsi.u_surface.derivative(1);
                    float duip1p1dphi = dsi.u_surface.derivative(2);
                    float duip1dtheta = dsi.u_surface.derivative(3);
                    float dvip1dui    = dsi.v_surface.derivative(0);
                    float dvip1dvi    = dsi.v_surface.derivative(1);
                    float dvip1dphi   = dsi.v_surface.derivative(2);
                    float dvip1dtheta = dsi.v_surface.derivative(3);

                    float dthetaip1dui    = theta_n.derivative(0);
                    float dthetaip1p1dvi  = theta_n.derivative(1);
                    float dthetaip1p1dphi = theta_n.derivative(2);
                    float dthetaip1dtheta = theta_n.derivative(3);
                    float dphiip1dui      = phi_n.derivative(0);
                    float dphiip1dvi      = phi_n.derivative(1);
                    float dphiip1dphi     = phi_n.derivative(2);
                    float dphiip1dtheta   = phi_n.derivative(3);

                    glm::mat4 Jray_ = glm::mat4(glm::vec4(duip1dui, dvip1dui, dphiip1dui, dthetaip1dui),
                                                glm::vec4(duip1p1dvi, dvip1dvi, dphiip1dvi, dthetaip1p1dvi),
                                                glm::vec4(duip1p1dphi, dvip1dphi, dphiip1dphi, dthetaip1p1dphi),
                                                glm::vec4(duip1dtheta, dvip1dtheta, dphiip1dtheta, dthetaip1dtheta));

                    // -------------------

                    // Jbsdf

                    glm::mat4x3 Jbsdf = glm::mat4x3(result.bsdf_weight.derivative(0),
                                                    result.bsdf_weight.derivative(1),
                                                    result.bsdf_weight.derivative(2),
                                                    result.bsdf_weight.derivative(3));

                    Jbsdf = Jbsdf * Jray;

                    // -------------------

                    Jray = Jray_ * Jray;

                    ray.JL -=
                        (diag(ray.radiance / (result.bsdf_weight.val() * result.sample_probability.val())) * Jbsdf +
                         diag(ray.throughput) * JLe);
                    auto Jrayinv    = glm::inverse(Jray);
                    glm::mat4x3 JL_ = ray.JL * Jrayinv;

                    // if(isnan(Jrayinv[0][0]))
                    // {
                    //     printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n------\n",
                    //            Jray[0][0],
                    //            Jray[0][1],
                    //            Jray[0][2],
                    //            Jray[0][3],
                    //            Jray[1][0],
                    //            Jray[1][1],
                    //            Jray[1][2],
                    //            Jray[1][3],
                    //            Jray[2][0],
                    //            Jray[2][1],
                    //            Jray[2][2],
                    //            Jray[2][3],
                    //            Jray[3][0],
                    //            Jray[3][1],
                    //            Jray[3][2],
                    //            Jray[3][3]);

                    //     printf("Theta: %f\n", theta_n.val());
                    // }

                    // ray.JL = JL - diag(ray.throughput) * JLe + diag(Le.val()) * Jb;
                    // Jb = diag(result.bsdf_value.val()) * Jb + diag(ray.throughput) * Jbsdf;

                    // 𝛿𝜋 += backward_grad(bsdf_value, 𝛿𝐿 ∗ 𝐿 / bsdf_value)
                    // = 1/pi * dL * L / (albedo / pi) = dL * L / albedo
                    glm::vec3 dLdbsdf = (ray.delta_y * (ray.radiance + 1e-4f)) / ((result.bsdf_weight.val()) + 1e-4f);
                    glm::vec2 dLdwo   = glm::zw(ray.delta_y * JL_);    // Only v2?

                    si.bsdf->sampleBSDFBackward(si, rng, dLdbsdf, dLdwo);

                    thrust::tie(u, v, phi, theta) =
                        CuDiff::make_variables<4>(dsi.u_surface.val(), dsi.v_surface.val(), phi_n.val(), theta_n.val());

                    auto sinTheta = CuDiff::sin(theta);

                    auto x = sinTheta * CuDiff::cos(phi);
                    auto y = CuDiff::cos(theta);
                    auto z = sinTheta * CuDiff::sin(phi);

                    auto origin = (1.0f - u - v) * dsi.P0 + u * dsi.P1 + v * dsi.P2;

                    ray.origin    = origin;
                    ray.direction = CuDiff::wrap(x, y, z);
                    ray.throughput *= result.bsdf_weight.val();
                    ray.valid = true;

                    last_si       = si;
                    last_bsdf_pdf = result.sample_probability;

                    if((int)(result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
                    {
                        last_si.valid = false;
                    }
                }
            }
        }
        else
        {
            if(params.environment_emitter)
            {
                bool mis_valid              = last_si.valid;
                float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
                float emitter_sampling_pdf =
                    mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) * emitter_selection_pdf
                              : 0.0f;
                float mis_weight = 1.0f;    // last_bsdf_pdf / (last_bsdf_pdf + emitter_sampling_pdf);
                ray.radiance -= mis_weight * ray.throughput * params.environment_emitter->evalLight(si);
            }
        }
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