#include "bsdf_functions.glsl"

subroutine vec3
sr_eval_brdf(vec3 base_color, float metallic, float roughness, vec3 normal, vec3 light_dir, vec3 view_dir);

subroutine(sr_eval_brdf) vec3
    eval_brdf_pbr(vec3 base_color, float metallic, float roughness, vec3 normal, vec3 light_dir, vec3 view_dir)
{
    vec3 H = normalize(light_dir + view_dir);

    float NdotH = max(dot(normal, H), 0.0);
    float NdotL = max(dot(normal, light_dir), 0.0);
    float NdotV = max(dot(normal, view_dir), 0.0);

    vec3 F0 = vec3(0.04);
    F0      = mix(F0, base_color, metallic);

    float NDF = distributionGGX(NdotH, roughness);
    float G   = geometrySmith(NdotL, NdotV, roughness);
    vec3 F    = fresnel_schlick(F0, max(dot(H, view_dir), 0.0));

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular     = numerator / denominator;

    vec3 kS            = F;
    vec3 kD            = vec3(1.0) - kS;
    vec3 color_diffuse = mix(base_color, vec3(0), metallic);

    vec3 brdf = specular + kD * color_diffuse / PI;
    return brdf;
}

subroutine uniform sr_eval_brdf eval_brdf;