float distributionGGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(float NdotL, float NdotV, float roughness)
{
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnel_schlick(const vec3 F0, const float VdotH)
{
    float p = clamp(1.0 - VdotH, 0.0, 1.0);
    return F0 + (1 - F0) * p * p * p * p * p;
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    float clamped = clamp(1.0 - cosTheta, 0.0, 1.0);
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * clamped * clamped * clamped * clamped * clamped;
}

vec3 eval_brdf(vec3 base_color, float metallic, float roughness, vec3 normal, vec3 light_dir, vec3 view_dir)
{
    vec3 H = normalize(light_dir + view_dir);

    float NdotH = max(dot(normal, H), 0.0);
    float NdotL = max(dot(normal, light_dir), 0.0);
    float NdotV = max(dot(normal, view_dir), 0.0);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, base_color, metallic);

    float NDF = distributionGGX(NdotH, roughness);
    float G = geometrySmith(NdotL, NdotV, roughness);
    vec3 F = fresnel_schlick(F0, max(dot(H, view_dir), 0.0));

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    vec3 color_diffuse = mix(base_color, vec3(0), metallic);

    vec3 brdf = specular + kD * color_diffuse / PI;
    return brdf;
}