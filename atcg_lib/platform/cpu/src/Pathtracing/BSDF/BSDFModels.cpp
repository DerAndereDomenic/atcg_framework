#include <Pathtracing/BSDF/BSDFModels.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
PBRBSDF::PBRBSDF(const Material& material)
{
    _diffuse_image   = material.getDiffuseTexture()->getData(atcg::CPU);
    _roughness_image = material.getRoughnessTexture()->getData(atcg::CPU);
    _metallic_image  = material.getMetallicTexture()->getData(atcg::CPU);
}

PBRBSDF::~PBRBSDF() {}

BSDFSamplingResult PBRBSDF::sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
{
    glm::vec3 diffuse_color = texture(_diffuse_image, si.uv);
    float metallic          = texture(_metallic_image, si.uv).x;
    float roughness         = texture(_roughness_image, si.uv).x;

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

    return samplePBR(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

BSDFEvalResult PBRBSDF::evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir)
{
    atcg::BSDFEvalResult result;

    glm::vec3 diffuse_color = texture(_diffuse_image, si.uv);
    float metallic          = texture(_metallic_image, si.uv).x;
    float roughness         = texture(_roughness_image, si.uv).x;
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    return evalPBR(si, outgoing_dir, diffuse_color, metallic_color, roughness, metallic);
}

RefractiveBSDF::RefractiveBSDF(const Material& material)
{
    _diffuse_image = material.getDiffuseTexture()->getData(atcg::CPU);
    _ior           = material.ior;
}

RefractiveBSDF::~RefractiveBSDF() {}

BSDFSamplingResult RefractiveBSDF::sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
{
    glm::vec3 diffuse_color = texture(_diffuse_image, si.uv);

    return sampleRefractive(si, diffuse_color, _ior, rng);
}

BSDFEvalResult RefractiveBSDF::evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir)
{
    return evalRefractive(si, outgoing_dir);
}
}    // namespace atcg