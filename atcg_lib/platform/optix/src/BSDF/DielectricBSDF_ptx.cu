#pragma cuda_source_property_format = PTX

#include <Core/CUDA.h>

#include <Math/Random.h>

#include <Math/Functions.h>
#include <Core/SurfaceInteraction.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <BSDF/DielectricBSDFData.cuh>


namespace detail
{

/**
 * @brief Sample a refractive BSDF
 *
 * @param si The surface interaction
 * @param diffuse_color The diffuse color
 * @param ior The index of refraction
 * @param rng The rng
 *
 * @return The sampling result
 */
ATCG_HOST_DEVICE ATCG_FORCE_INLINE atcg::BSDFSamplingResult sampleRefractive(const atcg::SurfaceInteraction& si,
                                                                             const glm::vec3& reflectance_color,
                                                                             const float ior,
                                                                             atcg::PCG32& rng)
{
    // Determine surface parameters
    bool outsidein             = glm::dot(si.incoming_direction, si.normal) < 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta                  = outsidein ? 1.0f / ior : ior;

    // Compute outgoing ray directions
    glm::vec3 transmitted_ray_dir = glm::refract(si.incoming_direction, interface_normal, eta);
    glm::vec3 reflected_ray_dir   = glm::reflect(si.incoming_direction, interface_normal);

    // Fresnel reflectance at normal incidence
    float F0 = (eta - 1) / (eta + 1);
    F0       = F0 * F0;

    float NdotL = glm::abs(glm::dot(si.incoming_direction, interface_normal));

    // Reflection an transmission probabilities
    float reflection_probability   = atcg::fresnel_schlick(F0, NdotL);
    float transmission_probability = 1.0f - reflection_probability;
    if(glm::dot(transmitted_ray_dir, transmitted_ray_dir) < 1e-6f)
    {
        // Total internal reflection!
        transmission_probability = 0.0f;
        reflection_probability   = 1.0f;
    }


    // Compute sampling result
    atcg::BSDFSamplingResult result;
    result.sample_probability = 0;

    // Stochastically select a reflection or transmission via russian roulette
    if(rng.next1d() < reflection_probability)
    {
        // Select the reflection event
        // We sample the BDSF exactly.
        result.bsdf_weight        = reflectance_color;
        result.out_dir            = reflected_ray_dir;
        result.sample_probability = reflection_probability;
    }
    else
    {
        // Select the transmission event
        // We sample the BDSF exactly.
        result.bsdf_weight        = reflectance_color;
        result.out_dir            = transmitted_ray_dir;
        result.sample_probability = transmission_probability;
    }

    result.flags = atcg::BSDFComponentType::IdealReflection | atcg::BSDFComponentType::IdealReflection;

    return result;
}
}    // namespace detail

extern "C" __device__ atcg::BSDFSamplingResult
__direct_callable__sample_dielectricbsdf(const atcg::SurfaceInteraction& si, atcg::PCG32& rng)
{
    const atcg::DielectricBSDFData* sbt_data =
        *reinterpret_cast<const atcg::DielectricBSDFData**>(optixGetSbtDataPointer());

    float4 reflectance_u        = tex2D<float4>(sbt_data->diffuse_texture, si.uv.x, si.uv.y);
    glm::vec3 reflectance_color = glm::vec3(reflectance_u.x, reflectance_u.y, reflectance_u.z);

    return detail::sampleRefractive(si, reflectance_color, sbt_data->ior, rng);
}

extern "C" __device__ atcg::BSDFEvalResult __direct_callable__eval_dielectricbsdf(const atcg::SurfaceInteraction& si,
                                                                                  const glm::vec3& outgoing_dir)
{
    return atcg::BSDFEvalResult();
}