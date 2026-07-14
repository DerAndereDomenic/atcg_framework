uniform samplerCube irradiance_map;
uniform samplerCube prefilter_map;
uniform sampler2D lut;

subroutine vec3
sr_image_based_lighting(vec3 base_color, float metallic, float roughness, float ior, vec3 normal, vec3 view_dir);

subroutine(sr_image_based_lighting) vec3
    image_based_lighting_pbr(vec3 base_color, float metallic, float roughness, float ior, vec3 normal, vec3 view_dir)
{
    float eta    = 1.0 / ior;
    float F0_eta = (1 - eta) / (1 + eta);
    F0_eta *= F0_eta;
    vec3 F0            = vec3(F0_eta);
    F0                 = mix(F0, base_color, metallic);
    float NdotV        = max(dot(normal, view_dir), 0.0);
    vec3 kS            = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD            = 1.0 - kS;
    vec3 irradiance    = texture(irradiance_map, normal).rgb;
    vec3 color_diffuse = mix(base_color, vec3(0), metallic);
    vec3 diffuse       = irradiance * color_diffuse;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R                         = reflect(-view_dir, normal);
    vec3 prefilteredColor          = textureLod(prefilter_map, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 lutbrdf                   = texture(lut, vec2(NdotV, roughness)).rg;
    vec3 specular                  = prefilteredColor * (kS * lutbrdf.x + lutbrdf.y);
    vec3 ambient                   = (kD * diffuse + specular);

    return ambient;
}

subroutine(sr_image_based_lighting) vec3
    image_based_lighting_diffuse(vec3 base_color, float metallic, float roughness, float ior, vec3 normal, vec3 view_dir)
{
    vec3 irradiance    = texture(irradiance_map, normal).rgb;
    vec3 color_diffuse = base_color;
    vec3 diffuse       = irradiance * color_diffuse;
    vec3 ambient       = diffuse;

    return ambient;
}

subroutine(sr_image_based_lighting) vec3
    image_based_lighting_glass(vec3 base_color, float metallic, float roughness, float ior, vec3 normal, vec3 view_dir)
{
    float eta    = 1.0 / ior;
    float F0_eta = (1 - eta) / (1 + eta);
    F0_eta *= F0_eta;
    vec3 F0 = vec3(F0_eta);

    float NdotV = max(dot(normal, view_dir), 0.0);

    const float MAX_REFLECTION_LOD = 4.0;
    float kS                       = fresnel_schlick(F0, max(0, dot(normal, view_dir))).r;

    vec3 reflected = reflect(-view_dir, normal);
    vec3 refracted = refract(-view_dir, normal, eta);

    if(length(refracted) < 1e-6)
    {
        kS = 1.0;
    }

    vec3 prefiltered_reflected = textureLod(prefilter_map, reflected, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 lutbrdf               = texture(lut, vec2(NdotV, roughness)).rg;
    vec3 bsdf_reflected        = prefiltered_reflected * (kS * lutbrdf.x + lutbrdf.y);

    vec3 prefiltered_refracted = textureLod(prefilter_map, refracted, roughness * MAX_REFLECTION_LOD).rgb;
    vec3 bsdf_refracted        = prefiltered_refracted * ((1.0 - kS) * lutbrdf.x);

    vec3 ambient = base_color * (bsdf_reflected + bsdf_refracted);

    // R = refract(-view_dir, normal, eta);
    // if(length(R) > 1e-5) glass_color += (1.0 - reflection_prob) * textureLod(prefilter_map, R, 0).rgb;

    // return vec3(glass_color * base_color);
    return ambient;
}

subroutine(sr_image_based_lighting) vec3
    image_based_lighting_null(vec3 base_color, float metallic, float roughness, float ior, vec3 normal, vec3 view_dir)
{
    vec3 prefiltered_refracted = textureLod(prefilter_map, -view_dir, 0).rgb;

    return prefiltered_refracted;
}

subroutine uniform sr_image_based_lighting image_based_lighting;