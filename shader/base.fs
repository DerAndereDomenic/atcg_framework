#version 330 core

#define PI 3.14159
#define MAX_LIGHTS 32

layout (location = 0) out vec4 outColor;
layout (location = 1) out int outEntityID;

in vec3 frag_normal;
in vec3 frag_pos;
in vec3 frag_color;
in vec2 frag_uv;
in mat3 frag_tbn;

uniform vec3 camera_pos;
uniform vec3 camera_dir;
uniform vec3 flat_color;
uniform int entityID;
uniform int use_ibl;

// Material textures
uniform sampler2D texture_diffuse;
uniform sampler2D texture_normal;
uniform sampler2D texture_roughness;
uniform sampler2D texture_metallic;
uniform samplerCube irradiance_map;
uniform samplerCube prefilter_map;
uniform sampler2D lut;

// Light data
uniform vec3 light_colors[MAX_LIGHTS];
uniform float light_intensities[MAX_LIGHTS];
uniform vec3 light_positions[MAX_LIGHTS];
uniform samplerCube light_shadows[MAX_LIGHTS];
uniform int num_lights = 0;

// Constants over the shader
vec3 view_dir = vec3(0);
float NdotV = 0;
vec3 color_diffuse = vec3(0);
float roughness = 0.0;
float metallic = 0.0;
vec3 normal = vec3(0.0);
vec3 F0 = vec3(0.0);

float distributionGGX(float NdotH, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom/denom;
}

float geometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r)/8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom/denom;
}


float geometrySmith(float NdotL, float NdotV, float roughness)
{
	float ggx2 = geometrySchlickGGX(NdotV, roughness);
	float ggx1 = geometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

vec3 fresnel_schlick(const vec3 F0, const float VdotH)
{
    float p = clamp(1.0-VdotH, 0.0, 1.0);
	return F0 + (1 - F0) * p * p * p * p * p;
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    float clamped = clamp(1.0 - cosTheta, 0.0, 1.0);
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * clamped * clamped * clamped * clamped * clamped;
}

float shadowCalculation(int i, vec3 pos)
{
    // get vector between fragment position and light position
    vec3 fragToLight = pos - light_positions[i];
    // use the light to fragment vector to sample from the depth map    
    float closestDepth = texture(light_shadows[i], fragToLight).r;
    // it is currently in linear range between [0,1]. Re-transform back to original value
    float far_plane = 100.0;
    closestDepth *= far_plane;
    // now get current linear depth as the length between the fragment and light position
    float currentDepth = length(fragToLight);
    // now test for shadows
    float bias = 0.05; 
    float shadow = currentDepth -  bias > closestDepth ? 1.0 : 0.0;

    return shadow;
}

vec3 eval_brdf(vec3 light_dir)
{
    vec3 H = normalize(light_dir + view_dir);

    float NdotH = max(dot(normal, H), 0.0);
    float NdotL = max(dot(normal, light_dir), 0.0);

    float NDF = distributionGGX(NdotH, roughness);
    float G = geometrySmith(NdotL, NdotV, roughness);
    vec3 F = fresnel_schlick(F0, max(dot(H, view_dir), 0.0));

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = numerator/denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= (1.0 - metallic);

    vec3 brdf = specular + kD * color_diffuse / PI;
    return brdf;
}

void main()
{
    // Set globals
    view_dir = normalize(camera_pos - frag_pos);
    vec4 diffuse_lookup = texture(texture_diffuse, frag_uv);
    color_diffuse = frag_color * flat_color * diffuse_lookup.rgb;
    roughness = texture(texture_roughness, frag_uv).r;
    metallic = texture(texture_metallic, frag_uv).r;
    vec3 texture_normal = normalize(texture(texture_normal, frag_uv).rgb * 2.0 - 1.0);
    normal = frag_tbn * texture_normal;
    F0 = vec3(0.04);
	F0 = mix(F0, color_diffuse, metallic);
    NdotV = max(dot(normal, view_dir), 0.0);

    // Define colocated direction light
    vec3 light_radiance = vec3(2.0);
    vec3 light_dir = normalize(camera_dir);

    // PBR material shader
    vec3 view_light = vec3(0);
    if(num_lights <= 0)
    {
        vec3 brdf = eval_brdf(light_dir);
        float NdotL = max(dot(normal, light_dir), 0.0);
        view_light = brdf * light_radiance * NdotL;
    }

    // Point lights
    vec3 point_light_contribution = vec3(0);
    for(int i = 0; i < num_lights; ++i)
    {
        vec3 light_dir = light_positions[i] - frag_pos;
        float r = length(light_dir);
        light_dir = light_dir / r;

        vec3 pl_brdf = eval_brdf(light_dir);
        vec3 light_radiance = light_intensities[i] * light_colors[i] / (r * r);
        float NdotL = max(dot(normal, light_dir), 0.0);
        float shadow = shadowCalculation(i, frag_pos);
        point_light_contribution += (1.0 - shadow) * pl_brdf * NdotL * light_radiance;
    }

    // IBL
    vec3 kS = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD = 1.0 - kS;
    kD *= (1.0 - metallic);
    vec3 irradiance = texture(irradiance_map, normal).rgb;
    vec3 diffuse = irradiance * color_diffuse;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-view_dir, normal);
    vec3 prefilteredColor = textureLod(prefilter_map, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 lutbrdf = texture(lut, vec2(NdotV, roughness)).rg;
    vec3 H = normalize(light_dir + view_dir);
    vec3 F = fresnel_schlick(F0, max(dot(normal, view_dir), 0.0));
    vec3 specular = prefilteredColor * (F * lutbrdf.x + lutbrdf.y);
    vec3 ambient = (kD * diffuse + specular);

    vec3 color = (1.0 - float(use_ibl)) * view_light + (float(use_ibl)) * ambient + point_light_contribution;
    
    float frag_dist = length(camera_pos - frag_pos);
    outColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0/2.4)), diffuse_lookup.w *( 1.0 - pow(1.01, frag_dist - 1000)));
    outEntityID = entityID;
}