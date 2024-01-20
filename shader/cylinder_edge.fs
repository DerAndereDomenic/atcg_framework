#version 330 core

#define PI 3.14159

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

float distributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N,H), 0.0);
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


float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N,V), 0.0);
	float NdotL = max(dot(N,L), 0.0);
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
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
} 

void main()
{
    // Define colocated direction light
    vec3 light_radiance = vec3(2.0);
    vec3 light_dir = normalize(camera_dir);
    vec3 view_dir = normalize(camera_pos - frag_pos);

    // Get parameters
    vec4 diffuse_lookup = texture(texture_diffuse, frag_uv);
    vec3 color_diffuse = frag_color * flat_color * diffuse_lookup.rgb;
    float roughness = texture(texture_roughness, frag_uv).r;
    float metallic = texture(texture_metallic, frag_uv).r;
    vec3 texture_normal = normalize(texture(texture_normal, frag_uv).rgb * 2.0 - 1.0);

    // PBR material shader
    vec3 normal = frag_tbn * texture_normal;
    vec3 F0 = vec3(0.04);
	F0 = mix(F0, color_diffuse, metallic);

    vec3 H = normalize(light_dir + view_dir);
    float NDF = distributionGGX(normal, H, roughness);
    float G = geometrySmith(normal, view_dir, light_dir, roughness);
    vec3 F = fresnel_schlick(F0, max(dot(H, view_dir), 0.0));

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, view_dir), 0.0) * max(dot(normal, light_dir), 0.0) + 0.0001;
    vec3 specular = numerator/denominator;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= (1.0 - metallic);

    vec3 brdf = specular + kD * color_diffuse / PI;

    // IBL
    kS = fresnelSchlickRoughness(max(dot(normal, view_dir), 0.0), F0, roughness);
    kD = 1.0 - kS;
    kD *= (1.0 - metallic);
    vec3 irradiance = texture(irradiance_map, normal).rgb;
    vec3 diffuse = irradiance * color_diffuse;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-view_dir, normal);
    vec3 prefilteredColor = textureLod(prefilter_map, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 lutbrdf = texture(lut, vec2(max(dot(normal, view_dir), 0.0), roughness)).rg;
    specular = prefilteredColor * (F * lutbrdf.x + lutbrdf.y);
    vec3 ambient = (kD * diffuse + specular);

    vec3 color = (1.0 - float(use_ibl)) * brdf * light_radiance * max(0.0f, dot(normal, light_dir)) + (float(use_ibl)) * ambient;
    
    float frag_dist = length(camera_pos - frag_pos);
    outColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0/2.4)), diffuse_lookup.w * (1.0 - pow(1.01, frag_dist - 1000)));
    outEntityID = entityID;
}