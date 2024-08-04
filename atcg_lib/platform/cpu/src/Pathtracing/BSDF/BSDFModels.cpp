#include <Pathtracing/BSDF/BSDFModels.h>

namespace atcg
{

namespace detail
{
static glm::vec3 read_image(const torch::Tensor& image, const glm::vec2& uv)
{
    glm::vec4 result(0);

    uint32_t image_width  = image.size(1);
    uint32_t image_height = image.size(0);
    glm::ivec2 pixel(uv.x * image_width, image_height - uv.y * image_height);
    pixel = glm::clamp(pixel, glm::ivec2(0), glm::ivec2(image_width - 1, image_height - 1));

    glm::vec3 color;

    uint32_t channels = image.size(2);
    bool is_hdr       = image.scalar_type() == torch::kFloat32;
    if(is_hdr)
    {
        if(channels == 4 || channels == 3)
        {
            color = glm::vec3(image.index({pixel.y, pixel.x, 0}).item<float>(),
                              image.index({pixel.y, pixel.x, 1}).item<float>(),
                              image.index({pixel.y, pixel.x, 2}).item<float>());
        }
        else
        {
            color = glm::vec3(image.index({pixel.y, pixel.x, 0}).item<float>());
        }
    }
    else
    {
        if(channels == 4 || channels == 3)
        {
            glm::u8vec3 val = glm::u8vec3(image.index({pixel.y, pixel.x, 0}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 1}).item<uint8_t>(),
                                          image.index({pixel.y, pixel.x, 2}).item<uint8_t>());
            color           = glm::vec3((float)val.x, (float)val.y, (float)val.z) / 255.0f;
        }
        else
        {
            uint8_t val = image.index({pixel.y, pixel.x, 0}).item<uint8_t>();
            color       = glm::vec3((float)val) / 255.0f;
        }
    }

    return color;
}
}    // namespace detail

PBRBSDF::PBRBSDF(const Material& material)
{
    _diffuse_image   = material.getDiffuseTexture()->getData(atcg::CPU);
    _roughness_image = material.getRoughnessTexture()->getData(atcg::CPU);
    _metallic_image  = material.getMetallicTexture()->getData(atcg::CPU);
}

PBRBSDF::~PBRBSDF() {}

BSDFSamplingResult PBRBSDF::sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
{
    glm::vec3 diffuse_color = detail::read_image(_diffuse_image, si.uv);
    float metallic          = detail::read_image(_metallic_image, si.uv).x;
    float roughness         = detail::read_image(_roughness_image, si.uv).x;

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04) + metallic * diffuse_color;

    return sampleGGX(si, diffuse_color, metallic_color, metallic, roughness, rng);
}

BSDFEvalResult PBRBSDF::evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir)
{
    // TODO: Make this more generic
    atcg::BSDFEvalResult result;

    glm::vec3 diffuse_color = detail::read_image(_diffuse_image, si.uv);
    float metallic          = detail::read_image(_metallic_image, si.uv).x;
    float roughness         = detail::read_image(_roughness_image, si.uv).x;
    roughness = glm::max(roughness * roughness, 1e-3f);    // In the real time shaders, roughness is squared

    glm::vec3 metallic_color = (1.0f - metallic) * glm::vec3(0.04f) + metallic * diffuse_color;

    glm::vec3 light_dir = outgoing_dir;
    glm::vec3 view_dir  = -si.incoming_direction;

    glm::vec3 H = glm::normalize(light_dir + view_dir);

    float NdotH = glm::max(glm::dot(si.normal, H), 0.0f);
    float NdotV = glm::max(glm::dot(si.normal, view_dir), 0.0f);
    float NdotL = glm::max(glm::dot(si.normal, light_dir), 0.0f);

    if(NdotL <= 0.0f || NdotV <= 0.0f) return result;

    float NDF   = atcg::D_GGX(NdotH, roughness);
    float G     = atcg::geometrySmith(NdotL, NdotV, roughness);
    glm::vec3 F = atcg::fresnel_schlick(metallic_color, glm::max(glm::dot(H, view_dir), 0.0f));

    glm::vec3 numerator = NDF * G * F;
    float denominator   = 4.0 * NdotV * NdotL + 1e-5f;
    glm::vec3 specular  = numerator / denominator;

    glm::vec3 kS = F;
    glm::vec3 kD = glm::vec3(1.0) - kS;
    kD *= (1.0 - metallic);

    float diffuse_probability =
        glm::dot(diffuse_color, glm::vec3(1)) /
        (glm::dot(diffuse_color, glm::vec3(1)) + glm::dot(metallic_color, glm::vec3(1)) + 1e-5f);
    float specular_probability    = 1 - diffuse_probability;
    float diffuse_pdf             = NdotL / glm::pi<float>();
    float halfway_pdf             = NDF * NdotH;
    float halfway_to_outgoing_pdf = atcg::warp_normal_to_reflected_direction_pdf(outgoing_dir, H);    // 1 / (4*HdotV)
    float specular_pdf            = halfway_pdf * halfway_to_outgoing_pdf;

    result.bsdf_value         = specular + kD * diffuse_color / glm::pi<float>();
    result.sample_probability = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf + 1e-5f;

    return result;
}

RefractiveBSDF::RefractiveBSDF(const Material& material)
{
    _diffuse_image = material.getDiffuseTexture()->getData(atcg::CPU);
    _ior           = material.ior;
}

RefractiveBSDF::~RefractiveBSDF() {}

BSDFSamplingResult RefractiveBSDF::sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
{
    glm::vec3 diffuse_color = detail::read_image(_diffuse_image, si.uv);

    // Determine surface parameters
    bool outsidein             = glm::dot(si.incoming_direction, si.normal) < 0;
    glm::vec3 interface_normal = outsidein ? si.normal : -si.normal;
    float eta                  = outsidein ? 1.0f / _ior : _ior;

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
        result.bsdf_weight        = diffuse_color;
        result.out_dir            = reflected_ray_dir;
        result.sample_probability = reflection_probability;
    }
    else
    {
        // Select the transmission event
        // We sample the BDSF exactly.
        result.bsdf_weight        = diffuse_color;
        result.out_dir            = transmitted_ray_dir;
        result.sample_probability = transmission_probability;
    }

    return result;
}

BSDFEvalResult RefractiveBSDF::evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir)
{    // TODO: Return empty result for now because most of the time a randomly sampled direction will not hit the delta
    // function of a perfect reflection/transmission
    return atcg::BSDFEvalResult();
}
}    // namespace atcg