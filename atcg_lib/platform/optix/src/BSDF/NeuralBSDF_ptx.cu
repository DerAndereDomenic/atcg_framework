#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Utils/HostDevice.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/NeuralBSDFData.cuh>
#include <BSDF/Sampling.h>

#include <DataStructure/Frame.h>

inline __device__ glm::vec3 albedo(const atcg::NeuralBSDFData* sbt_data, const glm::vec2& uvs)
{
    using T_IN     = OptixCoopVec<half, 8>;
    using T_HIDDEN = OptixCoopVec<half, 64>;
    using T_OUT    = OptixCoopVec<half, 8>;

    T_IN input;
    input[0] = __float2half(uvs.x);
    input[1] = __float2half(uvs.y);
    input[2] = __float2half(glm::sin(glm::two_pi<float>() * uvs.x));
    input[3] = __float2half(glm::sin(glm::two_pi<float>() * uvs.y));
    input[4] = __float2half(glm::sin(2.0f * glm::two_pi<float>() * uvs.x));
    input[5] = __float2half(glm::sin(2.0f * glm::two_pi<float>() * uvs.y));
    input[6] = __float2half(glm::sin(4.0f * glm::two_pi<float>() * uvs.x));
    input[7] = __float2half(glm::sin(4.0f * glm::two_pi<float>() * uvs.y));

    auto result = sbt_data->_device_mlp->forward(input);

    glm::vec3 alb =
        1.0f / (1.0f + glm::exp(-glm::vec3(__half2float(result[0]), __half2float(result[1]), __half2float(result[2]))));

    if(!isfinite(alb.x) || !isfinite(alb.y) || !isfinite(alb.z)) printf("%f %f %f\n", alb.x, alb.y, alb.z);
    return alb;
}

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_neuralbsdf(const atcg::SurfaceInteraction& si,
                                     const atcg::SampledWavelengths& wavelengths,
                                     atcg::PCG32& rng)
{
    const atcg::NeuralBSDFData* sbt_data = *reinterpret_cast<const atcg::NeuralBSDFData**>(optixGetSbtDataPointer());
    atcg::BSDFSamplingResult result;

    float NdotV = glm::max(0.0f, glm::dot(si.normal, -si.incoming_direction));
    if(NdotV <= 0.0f)
    {
        return result;
    }

    atcg::SamplingStrategy<atcg::SamplingStrategyType::HEMISPHERE_COSINE> sampler;
    glm::vec3 local_dir = sampler.sample(rng.next2d());
    glm::vec3 world_dir = atcg::Frame(si.normal).toWorld(local_dir);
    float pdf           = sampler.pdf(local_dir);

    result.bsdf_weight        = atcg::SampledSpectrum(albedo(sbt_data, si.uv));
    result.flags              = atcg::BSDFComponentType::DiffuseReflection;
    result.out_dir            = world_dir;
    result.sample_probability = pdf;

    return result;
}

extern "C" __device__ atcg::BSDFEvalResult
__direct_callable__eval_neuralbsdf(const atcg::SurfaceInteraction& si,
                                   const glm::vec3& outgoing_dir,
                                   const atcg::SampledWavelengths& wavelengths)
{
    const atcg::NeuralBSDFData* sbt_data = *reinterpret_cast<const atcg::NeuralBSDFData**>(optixGetSbtDataPointer());
    atcg::BSDFEvalResult result;

    float NdotL = glm::max(0.0f, glm::dot(si.normal, outgoing_dir));
    float NdotV = glm::max(0.0f, glm::dot(si.normal, -si.incoming_direction));

    if(NdotL <= 0.0f || NdotV <= 0.0f)
    {
        return result;
    }

    result.bsdf_value         = atcg::SampledSpectrum(albedo(sbt_data, si.uv) / glm::pi<float>() * NdotL);
    result.sample_probability = NdotL / glm::pi<float>();
    result.flags              = atcg::BSDFComponentType::DiffuseReflection;

    return result;
}