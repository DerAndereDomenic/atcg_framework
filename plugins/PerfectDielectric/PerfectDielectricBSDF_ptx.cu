#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Utils/HostDevice.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include "PerfectDielectricData.cuh"
#include <BSDF/BSDFFunctions.h>
#include <BSDF/Sampling.h>

#include <DataStructure/Frame.h>

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_perfectdielectricbsdf(const atcg::SurfaceInteraction& si,
                                                const atcg::SampledWavelengths& wavelengths,
                                                atcg::PCG32& rng,
                                                const atcg::BSDFContext& ctx)
{
    const atcg::PerfectDielectricData* sbt_data =
        *reinterpret_cast<const atcg::PerfectDielectricData**>(optixGetSbtDataPointer());

    float ior = sbt_data->ior_texture.read(si.uv);

    glm::vec3 wi = -si.incoming_direction;

    // Determine surface parameters
    bool outsidein             = glm::dot(wi, si.normal) > 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta                  = outsidein ? 1.0f / ior : ior;

    atcg::Frame local_frame = atcg::Frame(interface_normal);

    // Compute outgoing ray directions
    glm::vec3 transmitted_ray_dir = glm::refract(-wi, interface_normal, eta);
    glm::vec3 reflected_ray_dir   = glm::reflect(-wi, interface_normal);

    // Reflection an transmission probabilities
    float NdotV                    = glm::dot(wi, interface_normal);
    float F                        = atcg::fresnel(eta, NdotV);
    float reflection_probability   = F;
    float transmission_probability = 1.0f - reflection_probability;
    if(glm::dot(transmitted_ray_dir, transmitted_ray_dir) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = 0.0f;
        reflection_probability   = 1.0f;
    }

    // Compute sampling result
    atcg::BSDFSamplingResult result;

    // Stochastically select a reflection or transmission via russian roulette
    if(rng.next1d() < reflection_probability)
    {
        result.bsdf_weight = atcg::SampledSpectrum(1.0f);
        result.pdf_dw      = reflection_probability;
        result.out_dir     = reflected_ray_dir;

        result.flags = atcg::BSDFComponentType::IdealReflection;
    }
    else
    {
        result.bsdf_weight = (ctx.transport_mode == atcg::TransportMode::RADIANCE) ? atcg::SampledSpectrum(eta * eta)
                                                                                   : atcg::SampledSpectrum(1.0f);

        result.pdf_dw  = transmission_probability;
        result.out_dir = transmitted_ray_dir;

        result.flags = atcg::BSDFComponentType::IdealTransmission;
    }

    return result;
}

extern "C" __device__ atcg::BSDFEvalResult
__direct_callable__eval_perfectdielectricbsdf(const atcg::SurfaceInteraction& si,
                                              const glm::vec3& outgoing_dir,
                                              const atcg::SampledWavelengths& wavelengths,
                                              const atcg::BSDFContext& ctx)
{
    const atcg::PerfectDielectricData* sbt_data =
        *reinterpret_cast<const atcg::PerfectDielectricData**>(optixGetSbtDataPointer());

    float ior = sbt_data->ior_texture.read(si.uv);

    atcg::BSDFEvalResult result;
    result.bsdf_value = atcg::SampledSpectrum(0);
    result.pdf_dw     = 0;
    result.flags      = atcg::BSDFComponentType::IdealReflection | atcg::BSDFComponentType::IdealTransmission;
    return result;
}

extern "C" __device__ float
__direct_callable__evalpdf_perfectdielectricbsdf(const atcg::SurfaceInteraction& si,
                                                 const glm::vec3& outgoing_dir,
                                                 const atcg::SampledWavelengths& wavelengths,
                                                 const atcg::BSDFContext& ctx)
{
    const atcg::PerfectDielectricData* sbt_data =
        *reinterpret_cast<const atcg::PerfectDielectricData**>(optixGetSbtDataPointer());

    return 0.0f;
}