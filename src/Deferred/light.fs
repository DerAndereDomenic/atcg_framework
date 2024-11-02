#version 330 core

#define PI 3.14159

layout (location = 0) out vec4 outColor;
layout (location = 1) out int outEntityID;

in vec2 frag_uv;

uniform vec3 camera_pos;
uniform vec3 camera_dir;
uniform int use_ibl;

// Material textures
uniform sampler2D position_texture;
uniform sampler2D normal_texture;
uniform sampler2D color_texture;
uniform sampler2D spec_met_texture;
uniform isampler2D entity_texture;
uniform samplerCube irradiance_map;
uniform samplerCube prefilter_map;
uniform sampler2D lut;

// Light data
uniform vec3 light_positions[32];
uniform vec3 light_colors[32];
uniform float light_intensities[32];
uniform int num_lights;

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

void main()
{
    vec3 frag_pos = texture(position_texture, frag_uv).rgb;

    // Get parameters
    vec3 color_diffuse = texture(color_texture, frag_uv).rgb;
    vec2 spec_metallic = texture(spec_met_texture, frag_uv).rg;
    float roughness = spec_metallic.y;
    float metallic = spec_metallic.x;
    vec3 normal = texture(normal_texture, frag_uv).rgb;
    if(length(normal) <= 1e-5) discard;
    vec3 color = vec3(0);

    for(int i = 0; i < num_lights; ++i)
    {
        // Define colocated direction light
        vec3 light_dir = normalize(light_positions[i] - frag_pos);
        float light_distance = length(light_positions[i] - frag_pos);
        vec3 light_radiance = light_intensities[i] * light_colors[i] / (light_distance * light_distance);
        vec3 view_dir = normalize(camera_pos - frag_pos);

        // PBR material shader
        vec3 F0 = vec3(0.04);
        F0 = mix(F0, color_diffuse, metallic);

        vec3 H = normalize(light_dir + view_dir);

        float NdotH = max(dot(normal, H), 0.0);
        float NdotV = max(dot(normal, view_dir), 0.0);
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
        color += brdf * light_radiance * NdotL;
    }

    // IBL
    // kS = fresnelSchlickRoughness(NdotV, F0, roughness);
    // kD = 1.0 - kS;
    // kD *= (1.0 - metallic);
    // vec3 irradiance = texture(irradiance_map, normal).rgb;
    // vec3 diffuse = irradiance * color_diffuse;

    // const float MAX_REFLECTION_LOD = 4.0;
    // vec3 R = reflect(-view_dir, normal);
    // vec3 prefilteredColor = textureLod(prefilter_map, R, roughness * MAX_REFLECTION_LOD).rgb;
    // vec2 lutbrdf = texture(lut, vec2(NdotV, roughness)).rg;
    // specular = prefilteredColor * (F * lutbrdf.x + lutbrdf.y);
    // vec3 ambient = (kD * diffuse + specular);

    
    outColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0/2.4)), 1);
    ivec2 pixel_coords = ivec2(frag_uv * textureSize(color_texture, 0));
    outEntityID = texelFetch(entity_texture, pixel_coords, 0).r;
}