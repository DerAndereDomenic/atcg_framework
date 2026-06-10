#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include "VolAttachedDiffPathtracingData.cuh"

#include <Core/TraceParameters.h>
#include <Core/SurfaceInteraction.h>
#include <Core/Payload.h>
#include <Math/Random.h>
#include <Utils/HostDevice.h>
#include <DataStructure/Frame.h>
#include <Integrator/MIS.h>
#include <Utils/HostDevice.h>
#include <Medium/MediumVPtrTable.cuh>

#include <CuDiff/CuDiff.h>
#include <CuDiff/ext/glm.h>

extern "C"
{
    __constant__ atcg::VolAttachedDiffPathtracingParams params;
}

struct RayContext
{
    bool valid;
    atcg::AnyInteraction ai0;
    atcg::AnyInteraction ai1;
    CuDiff::Dual<6, glm::vec3> last_normal;
    CuDiff::Dual<6, glm::vec2> last_uv;

    glm::vec3 throughput;
    glm::vec3 radiance;

    glm::vec3 delta_y;
    atcg::mat6x3 JL;

    const atcg::MediumVPtrTable* current_medium = nullptr;
};

struct DualInteractionResult
{
    atcg::AnyDualInteraction dai;
    CuDiff::Dual<6, glm::vec3> x0;
    CuDiff::Dual<6, glm::vec3> x1;
    CuDiff::Dual<6, glm::vec3> w;
};

struct DirectionSampleResult
{
    CuDiff::Dual<6, glm::vec3> out_dir;
    glm::vec3 bsdf_weight;
    atcg::mat6x3 dbsdf_dx0x1;
    float sample_probability;
};

struct NextVertexResult
{
    atcg::AnyInteraction next_ai;
    CuDiff::Dual<6, glm::vec3> next_normal;    // only meaningful if surface
    CuDiff::Dual<6, glm::vec2> next_uv;
    atcg::mat6 Jray_;
    glm::mat3 dxdw;
    glm::vec3 transmittance_weight;
    atcg::mat6x3 dtransmittance_dx0x1;
    bool valid;
};

ATCG_INLINE ATCG_DEVICE DualInteractionResult buildDualInteraction(const RayContext& ray)
{
    DualInteractionResult result;

    atcg::AnyInteraction ai0 = ray.ai0;
    atcg::AnyInteraction ai1 = ray.ai1;

    auto [x0, x1] = CuDiff::make_variables<6>(ai0->position, ai1->position);
    auto distance = CuDiff::length(x1 - x0);
    auto w        = (x1 - x0) / CuDiff::max(distance, 1e-5f);

    atcg::AnyDualInteraction dai;
    if(ai1.is_surface())
    {
        atcg::DualSurfaceInteraction dsi;
        dsi.position           = x1;
        dsi.incoming_direction = w;
        dsi.incoming_distance  = distance;
        dsi.normal             = ray.last_normal;
        dsi.uv                 = ray.last_uv;
        dai                    = dsi;
    }
    else
    {
        atcg::DualMediumInteraction dmi;
        dmi.position           = x1;
        dmi.incoming_direction = w;
        dmi.incoming_distance  = distance;
        dai                    = dmi;
    }

    result.dai = dai;
    result.x0  = x0;
    result.x1  = x1;
    result.w   = w;
    return result;
}

ATCG_INLINE ATCG_DEVICE atcg::mat6x3 handleDirectIlluminationSurface(RayContext& ray,
                                                                     atcg::AnyInteraction& ai0,
                                                                     atcg::AnyInteraction& ai1,
                                                                     atcg::AnyDualInteraction& dai,
                                                                     const atcg::mat6& Jray,
                                                                     const atcg::mat6x3& Jb,
                                                                     const atcg::SampledWavelengths& wavelengths,
                                                                     atcg::PCG32& rng)
{
    atcg::mat6x3 JLe = atcg::mat6x3(0.0f);

    atcg::SurfaceInteraction& si1     = ai1;
    atcg::DualSurfaceInteraction& dsi = dai;

    // ── Emitter hit (MIS-weighted) ──────────────────────────────────────────
    if(si1.emitter)
    {
        bool mis_valid              = ai0->isValid();
        float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
        float emitter_sampling_pdf =
            atcg::select(mis_valid, si1.emitter->evalLightSamplingPdf(ai0, ai1) * emitter_selection_pdf, 0.0f);
        float mis_weight  = atcg::PowerHeuristic<1>::apply(ai0->pdf, emitter_sampling_pdf);
        auto light_result = si1.emitter->evalLightForward(dsi, wavelengths);
        glm::vec3 Le      = mis_weight * light_result.radiance_weight_at_receiver;

        if(params.diff_mode == atcg::DiffMode::FORWARD)
        {
            ray.radiance += ray.throughput * Le;
        }
        else
        {
            ray.radiance -= ray.throughput * Le;
        }

        JLe = light_result.dLe_dx0x1 * Jray;

        if(params.diff_mode == atcg::DiffMode::FORWARD)
        {
            ray.JL += atcg::diag(ray.throughput) * JLe + atcg::diag(Le) * Jb;
        }
    }

    // ── Next-event estimation ───────────────────────────────────────────────
    do
    {
        if(params.num_emitters == 0) break;
        if(!si1.bsdf) break;

        uint32_t emitter_index                = rng.nextUint32() % params.num_emitters;
        float emitter_selection_pdf           = 1.0f / ((float)params.num_emitters);
        const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

        if(si1.emitter == emitter) break;

        atcg::EmitterDualSamplingResult emitter_sampling = emitter->sampleLightForward(dsi, wavelengths, rng);

        if(emitter_sampling.sampling_pdf == 0) break;

        emitter_sampling.sampling_pdf *= emitter_selection_pdf;
        emitter_sampling.radiance_weight_at_receiver =
            emitter_sampling.radiance_weight_at_receiver / emitter_selection_pdf;
        emitter_sampling.dLe_dx0x1 = emitter_sampling.dLe_dx0x1 / emitter_selection_pdf;

        bool occluded = traceOcclusion(params.handle,
                                       si1.position,
                                       emitter_sampling.direction_to_light,
                                       1e-3f,
                                       emitter_sampling.distance_to_light - 1e-3f,
                                       params.occlusion_trace_params);
        if(occluded) break;

        atcg::BSDFDualEvalResult bsdf_result =
            si1.bsdf->evalBSDFForward(dsi, emitter_sampling.direction_to_light, wavelengths);

        float bsdf_pdf   = atcg::select((int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0 ||
                                            (int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0,
                                        0.0f,
                                        bsdf_result.sample_probability);
        float mis_weight = atcg::PowerHeuristic<1>::apply(emitter_sampling.sampling_pdf, bsdf_pdf);

        glm::vec3 throughput_nee = ray.throughput * bsdf_result.bsdf_value;
        glm::vec3 radiance_nee   = mis_weight * throughput_nee * emitter_sampling.radiance_weight_at_receiver;

        if(params.diff_mode == atcg::DiffMode::FORWARD)
        {
            ray.radiance += radiance_nee;
        }
        else
        {
            ray.radiance -= radiance_nee;
        }

        auto JLe_nee   = emitter_sampling.dLe_dx0x1 * Jray;
        auto Jbsdf_nee = bsdf_result.dbsdf_dx0x1 * Jray;

        if(params.diff_mode == atcg::DiffMode::FORWARD)
        {
            atcg::mat6x3 Jb_nee = atcg::diag(bsdf_result.bsdf_value) * Jb + atcg::diag(ray.throughput) * Jbsdf_nee;

            ray.JL += mis_weight * (atcg::diag(emitter_sampling.radiance_weight_at_receiver) * Jb_nee +
                                    atcg::diag(throughput_nee) * JLe_nee);
        }
        else
        {
            ray.JL -= (atcg::diag(radiance_nee / bsdf_result.bsdf_value) * Jbsdf_nee +
                       mis_weight * atcg::diag(throughput_nee) * JLe_nee);

            glm::vec3 grad_out = (ray.delta_y * (radiance_nee + 1e-4f)) / (glm::vec3(bsdf_result.bsdf_value) + 1e-4f);
            si1.bsdf->evalBSDFBackward(si1, emitter_sampling.direction_to_light.val(), grad_out);
        }
    } while(false);

    return JLe;
}

ATCG_INLINE ATCG_DEVICE void handleDirectIlluminationMedium(RayContext& ray,
                                                            atcg::AnyInteraction& ai0,
                                                            atcg::AnyInteraction& ai1,
                                                            atcg::AnyDualInteraction& dai,
                                                            const atcg::mat6& Jray,
                                                            const atcg::mat6x3& Jb,
                                                            const atcg::SampledWavelengths& wavelengths,
                                                            atcg::PCG32& rng)
{
    // Nee for volumes
    atcg::MediumInteraction& mi      = ai1;
    atcg::DualMediumInteraction& dmi = dai;

    // NEE
    do
    {
        if(params.num_emitters == 0) break;

        uint32_t emitter_index = rng.nextUint32() % params.num_emitters;

        float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);

        const atcg::EmitterVPtrTable* emitter = params.emitters[emitter_index];

        atcg::EmitterDualSamplingResult emitter_sampling = emitter->sampleLightForward(dmi, wavelengths, rng);

        if(emitter_sampling.sampling_pdf == 0)
        {
            break;
        }
        emitter_sampling.sampling_pdf *= emitter_selection_pdf;
        emitter_sampling.radiance_weight_at_receiver /= emitter_selection_pdf;
        emitter_sampling.dLe_dx0x1 = emitter_sampling.dLe_dx0x1 / emitter_selection_pdf;

        atcg::DualSurfaceInteraction dsi_dummy;
        dsi_dummy.incoming_position  = dmi.position;
        dsi_dummy.incoming_direction = emitter_sampling.direction_to_light;

        atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                                 mi.position,
                                                                 emitter_sampling.direction_to_light.val(),
                                                                 0.0f,
                                                                 1e16f,
                                                                 &dsi_dummy,
                                                                 params.dual_trace_params);

        if(!dsi_dummy.isValid())
        {
            // Should not happen because we are inside the geometry
            break;
        }

        if(!dsi_dummy.bsdf || (int)(dsi_dummy.bsdf->flags & atcg::BSDFComponentType::NullTransmission) == 0)
        {
            break;
        }

        bool occluded = traceOcclusion(params.handle,
                                       dsi_dummy.position.val(),
                                       emitter_sampling.direction_to_light.val(),
                                       1e-3f,
                                       emitter_sampling.distance_to_light - dsi_dummy.incoming_distance.val() - 1e-3f,
                                       params.occlusion_trace_params);

        if(occluded)
        {
            break;
        }

        atcg::PCG32 rng_copy = rng;
        atcg::DualTransmittanceEvalResult transmittance_result =
            ray.current_medium->evalTransmittanceForward(dmi.position,
                                                         emitter_sampling.direction_to_light,
                                                         dsi_dummy.incoming_distance,
                                                         rng);

        auto phase_result =
            ray.current_medium->phase_function->evalPhaseFunctionForward(dmi, emitter_sampling.direction_to_light);
        float phase_pdf = phase_result.sampling_pdf;
        float sampling_pdf =
            atcg::select((int)(emitter->flags & atcg::EmitterFlags::InfinitesimalSize) != 0, 0.0f, phase_pdf);

        float mis_weight = atcg::BalanceHeuristic::apply(emitter_sampling.sampling_pdf, sampling_pdf);

        float weight             = transmittance_result.transmittance * phase_result.phase_function_value;
        glm::vec3 throughput_nee = ray.throughput * weight;
        glm::vec3 radiance_nee   = mis_weight * throughput_nee * emitter_sampling.radiance_weight_at_receiver;

        if(params.diff_mode == atcg::DiffMode::FORWARD)
        {
            ray.radiance += radiance_nee;
        }
        else
        {
            ray.radiance -= radiance_nee;
        }

        auto JLe_nee        = emitter_sampling.dLe_dx0x1 * Jray;
        auto Jphase         = phase_result.dphase_dx0x1 * Jray;
        auto Jtransmittance = transmittance_result.dtransmittance_dx0x1 * Jray;

        auto Jweight_nee = atcg::diag(phase_result.phase_function_value) * Jtransmittance +
                           atcg::diag(transmittance_result.transmittance) * Jphase;

        if(params.diff_mode == atcg::DiffMode::FORWARD)
        {
            atcg::mat6x3 Jb_nee = atcg::diag(weight) * Jb + atcg::diag(ray.throughput) * Jweight_nee;

            ray.JL += mis_weight * (atcg::diag(emitter_sampling.radiance_weight_at_receiver) * Jb_nee +
                                    atcg::diag(throughput_nee) * JLe_nee);
        }
        else
        {
            ray.JL -=
                (atcg::diag(radiance_nee / weight) * Jweight_nee + mis_weight * atcg::diag(throughput_nee) * JLe_nee);

            glm::vec3 grad_out = ray.delta_y * radiance_nee;

            ray.current_medium->evalTransmittanceBackward(mi.position,
                                                          emitter_sampling.direction_to_light.val(),
                                                          dsi_dummy.incoming_distance.val(),
                                                          rng_copy,
                                                          grad_out);

            ray.current_medium->phase_function->evalPhaseFunctionBackward(mi,
                                                                          emitter_sampling.direction_to_light.val(),
                                                                          grad_out);
        }


    } while(false);
}

ATCG_INLINE ATCG_DEVICE DirectionSampleResult sampleDirection(RayContext& ray,
                                                              atcg::AnyInteraction& ai0,
                                                              atcg::AnyInteraction& ai1,
                                                              atcg::AnyDualInteraction& dai,
                                                              const atcg::SampledWavelengths& wavelengths,
                                                              atcg::PCG32& rng)
{
    DirectionSampleResult result;

    if(ai1.is_surface())
    {
        atcg::SurfaceInteraction& si1     = ai1;
        atcg::DualSurfaceInteraction& dsi = dai;

        auto bsdf_result = si1.bsdf->sampleBSDFForward(dsi, wavelengths, rng);

        result.sample_probability = bsdf_result.sample_probability;
        result.out_dir            = bsdf_result.out_dir;
        result.bsdf_weight        = bsdf_result.bsdf_weight;
        result.dbsdf_dx0x1        = bsdf_result.dbsdf_dx0x1;

        // Medium transition: only change medium on transmission
        float cos_theta_curr_ray = glm::dot(si1.normal, si1.incoming_direction);
        float cos_theta_next_ray = glm::dot(si1.normal, result.out_dir.val());
        if(cos_theta_curr_ray * cos_theta_next_ray > 0)
        {
            ray.current_medium = cos_theta_next_ray < 0 ? si1.inside_medium : si1.outside_medium;
        }

        if((int)(bsdf_result.flags & atcg::BSDFComponentType::NullTransmission) == 0)
        {
            si1.pdf = result.sample_probability;

            if((int)(bsdf_result.flags & atcg::BSDFComponentType::AnyDelta) != 0)
            {
                ai1->setInvalid();
            }
        }
        else
        {
            // Propagate pdf for null transmission to the next vertex, so that it can be used for MIS weighting of
            // emitter sampling at the next vertex
            si1.pdf = ai0->pdf;
        }
    }
    else
    {
        atcg::DualMediumInteraction& dmi = dai;
        auto phase_result                = ray.current_medium->phase_function->samplePhaseFunctionForward(dmi, rng);

        result.sample_probability = phase_result.sampling_pdf;
        result.out_dir            = phase_result.outgoing_ray_dir;
        result.bsdf_weight        = glm::vec3(phase_result.phase_function_weight);
        result.dbsdf_dx0x1        = phase_result.dphase_dx0x1;

        ai1->pdf = result.sample_probability;
    }

    return result;
}

ATCG_INLINE ATCG_DEVICE NextVertexResult traceNextVertex(RayContext& ray,
                                                         const CuDiff::Dual<6, glm::vec3>& x1,
                                                         const CuDiff::Dual<6, glm::vec3>& out_dir,
                                                         const atcg::SampledWavelengths& wavelengths,
                                                         atcg::PCG32& rng)
{
    NextVertexResult result;
    result.transmittance_weight = glm::vec3(1.0f);
    result.dtransmittance_dx0x1 = atcg::mat6x3(0.0f);
    result.valid                = false;

    atcg::DualSurfaceInteraction next_dsi;
    next_dsi.incoming_position  = x1;
    next_dsi.incoming_direction = out_dir;

    atcg::traceWithDataPointer<atcg::DualSurfaceInteraction>(params.handle,
                                                             x1.val(),
                                                             out_dir.val(),
                                                             0.001f,
                                                             1e16f,
                                                             &next_dsi,
                                                             params.dual_trace_params);
    if(!next_dsi.isValid())
    {
        return result;
    }

    result.next_ai     = next_dsi.toSi();
    result.Jray_       = next_dsi.dx1x2_dx0x1;
    result.next_normal = next_dsi.normal;
    result.next_uv     = next_dsi.uv;
    result.dxdw        = next_dsi.dxdw;

    if(ray.current_medium)
    {
        float max_distance = next_dsi.incoming_distance.val();
        auto medium_result = ray.current_medium->sampleMediumEventForward(x1, out_dir, max_distance, wavelengths, rng);

        result.transmittance_weight = medium_result.transmittance_weight;
        result.dtransmittance_dx0x1 = medium_result.dtransmittance_dx0x1;

        if(medium_result.interaction.isValid())
        {
            // TODO There might be a transmittance weight + derivative even if medium sampling fails?
            result.next_ai = medium_result.interaction.toMi();
            result.Jray_   = medium_result.interaction.dx1x2_dx0x1;
            result.dxdw    = medium_result.interaction.dxdw;
        }
    }

    result.valid = true;
    return result;
}

ATCG_INLINE ATCG_DEVICE glm::mat3 getProjection(const atcg::AnyInteraction& ai)
{
    if(ai.is_surface())
    {
        return glm::mat3(1.0f) - glm::outerProduct(ai->reference_frame.localZ(), ai->reference_frame.localZ());
    }
    else
    {
        return glm::mat3(1.0f);
    }
}

ATCG_INLINE ATCG_DEVICE void updateDerivatives(RayContext& ray,
                                               atcg::AnyInteraction& ai0,
                                               atcg::AnyInteraction& ai1,
                                               atcg::AnyInteraction& next_ai,
                                               const NextVertexResult& next,
                                               const DirectionSampleResult& dir,
                                               const atcg::mat6x3& Jweight,
                                               const atcg::mat6x3& JLe,
                                               atcg::mat6& Jray,
                                               atcg::mat6x3& Jb,
                                               const atcg::SampledWavelengths& wavelengths,
                                               atcg::PCG32& rng_direction,
                                               atcg::PCG32& rng_position)
{
    auto P0 = getProjection(ai0);
    auto P1 = getProjection(ai1);
    auto P2 = getProjection(next_ai);

    auto P = atcg::mat6(P0, glm::mat3(0.0f), glm::mat3(0.0f), P1);
    auto Q = atcg::mat6(P1, glm::mat3(0.0f), glm::mat3(0.0f), P2);

    Jray = (Q * next.Jray_ * P) * Jray;
    Jray += atcg::mat6(0.01f * glm::sign(rng_direction.nextFloat() - 0.5f));    // Regularization

    if(params.diff_mode == atcg::DiffMode::FORWARD)
    {
        Jb = atcg::diag(dir.bsdf_weight * next.transmittance_weight) * Jb + atcg::diag(ray.throughput) * Jweight;
        return;
    }

    // ── Backward pass ────────────────────────────────────────────────────────
    ray.JL -= (atcg::diag(ray.radiance / (dir.bsdf_weight * next.transmittance_weight)) * Jweight +
               atcg::diag(ray.throughput) * JLe);

    auto Jray_inv    = atcg::pseudoinverse(Jray);
    atcg::mat6x3 JL_ = ray.JL * Jray_inv;

    if(ai1.is_surface())
    {
        atcg::SurfaceInteraction& si1 = ai1;
        if(!si1.bsdf) return;

        // 𝛿𝜋 += backward_grad(bsdf_value, 𝛿𝐿 ∗ 𝐿 / bsdf_value)
        glm::vec3 dL_dbsdf = (ray.delta_y * (ray.radiance + 1e-4f)) / (dir.bsdf_weight + 1e-4f);

        auto dL_dx2 = JL_.m01;
        auto dL_dwo = ray.delta_y * dL_dx2 * next.dxdw;

        si1.bsdf->sampleBSDFBackward(si1, rng_direction, dL_dbsdf, dL_dwo);
    }
    else /*if (ai1.is_medium())*/
    {
        // Medium backward
        atcg::MediumInteraction& mi1 = ai1;

        glm::vec3 dL_dphase = (ray.delta_y * (ray.radiance + 1e-4f)) / (dir.bsdf_weight + 1e-4f);

        auto dL_dx2 = JL_.m01;
        auto dL_dwo = ray.delta_y * dL_dx2 * next.dxdw;
        ray.current_medium->phase_function->samplePhaseFunctionBackward(mi1, rng_direction, dL_dphase, dL_dwo);
    }

    if(ray.current_medium)
    {
        // Medium transmittance backward
        float max_distance = next_ai->incoming_distance;
        glm::vec3 dLdw     = (ray.delta_y * (ray.radiance + 1e-4f)) / (next.transmittance_weight + 1e-4f);

        glm::vec3 dLdx2 = ray.delta_y * JL_.m01;
        ray.current_medium->sampleMediumEventFullBackward(ai1->position,
                                                          dir.out_dir.val(),
                                                          max_distance,
                                                          wavelengths,
                                                          rng_position,
                                                          dLdw,
                                                          dLdx2);
    }
}

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
    auto init_si1   = dsi1.toSi();
    ray.last_normal = dsi1.normal;
    ray.last_uv     = dsi1.uv;


    if(init_si1.isValid())
    {
        ray.valid = true;
    }

    ray.ai0      = si0;
    ray.ai0->pdf = 1.0f;
    ray.ai1      = init_si1;

    for(int n = 0; n < 512; ++n)
    {
        if(!ray.valid) break;
        ray.valid = false;

        float rr_prob = glm::max(glm::max(ray.throughput.r, ray.throughput.g), ray.throughput.b);
        if(rng.nextFloat() < rr_prob)
        {
            ray.throughput /= rr_prob;
        }
        else
        {
            break;
        }

        atcg::AnyInteraction ai0 = ray.ai0;
        atcg::AnyInteraction ai1 = ray.ai1;

        auto dual                     = buildDualInteraction(ray);
        atcg::AnyDualInteraction& dai = dual.dai;

        atcg::mat6x3 JLe = atcg::mat6x3(0.0f);
        if(ai1.is_surface())
        {
            JLe = handleDirectIlluminationSurface(ray, ai0, ai1, dai, Jray, Jb, wavelengths, rng);
        }
        else
        {
            handleDirectIlluminationMedium(ray, ai0, ai1, dai, Jray, Jb, wavelengths, rng);
        }

        if(ai1.is_surface() && !((atcg::SurfaceInteraction&)ai1).bsdf)
        {
            continue;
        }

        atcg::PCG32 rng_direction = rng;
        auto dir                  = sampleDirection(ray, ai0, ai1, dai, wavelengths, rng);

        if(dir.sample_probability <= 0.0f) continue;


        atcg::PCG32 rng_position = rng;
        auto next                = traceNextVertex(ray, dual.x1, dir.out_dir, wavelengths, rng_position);
        if(!next.valid) continue;

        auto Jweight = atcg::diag(next.transmittance_weight) * dir.dbsdf_dx0x1 +
                       atcg::diag(dir.bsdf_weight) * next.dtransmittance_dx0x1;
        Jweight      = Jweight * Jray;

        updateDerivatives(ray,
                          ai0,
                          ai1,
                          next.next_ai,
                          next,
                          dir,
                          Jweight,
                          JLe,
                          Jray,
                          Jb,
                          wavelengths,
                          rng_direction,
                          rng_position);

        ray.ai0         = ai1;
        ray.ai1         = next.next_ai;
        ray.last_normal = next.next_normal;
        ray.last_uv     = next.next_uv;

        ray.throughput *= dir.bsdf_weight * next.transmittance_weight;
        ray.valid = next.next_ai->isValid();

        // TODO
        // else
        // {
        //     if(params.environment_emitter)
        //     {
        //         bool mis_valid              = last_si.valid;
        //         float emitter_selection_pdf = 1.0f / ((float)params.num_emitters);
        //         float emitter_sampling_pdf =
        //             mis_valid ? params.environment_emitter->evalLightSamplingPdf(last_si, si) *
        //             emitter_selection_pdf
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