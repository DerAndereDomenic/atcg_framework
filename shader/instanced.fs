#version 400 core

#include "common/defines.glsl"
#include "common/bsdf.glsl"
#include "common/ibl.glsl"

layout(location = 0) out vec4 outColor;
layout(location = 1) out int outEntityID;

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
uniform sampler2D texture_emissive;
uniform float emission_scale;
uniform int emissive = 0;

// Light data
uniform vec3 light_colors[MAX_LIGHTS];
uniform float light_intensities[MAX_LIGHTS];
uniform vec3 light_positions[MAX_LIGHTS];
uniform samplerCubeArray shadow_maps;
uniform int num_lights = 0;
uniform int receive_shadow;

float shadowCalculation(int i, vec3 pos)
{
    // get vector between fragment position and light position
    vec3 fragToLight = pos - light_positions[i];
    // use the light to fragment vector to sample from the depth map    
    float closestDepth = texture(shadow_maps, vec4(fragToLight, float(i))).r;
    // it is currently in linear range between [0,1]. Re-transform back to original value
    float far_plane = 100.0;
    closestDepth *= far_plane;
    // now get current linear depth as the length between the fragment and light position
    float currentDepth = length(fragToLight);
    // now test for shadows
    float bias = 0.05;
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;

    return receive_shadow * shadow;
}

void main()
{
    // Set globals
    vec3 view_dir = normalize(camera_pos - frag_pos);
    vec4 diffuse_lookup = texture(texture_diffuse, frag_uv);
    vec3 base_color = frag_color * flat_color * diffuse_lookup.rgb;
    float roughness = texture(texture_roughness, frag_uv).r;
    float metallic = texture(texture_metallic, frag_uv).r;
    vec3 texture_normal = normalize(texture(texture_normal, frag_uv).rgb * 2.0 - 1.0);
    vec3 normal = frag_tbn * texture_normal;

    // Define colocated direction light
    vec3 light_radiance = vec3(2.0);
    vec3 light_dir = normalize(camera_dir);

    // PBR material shader
    vec3 view_light = vec3(0);
    if(num_lights <= 0)
    {
        vec3 brdf = eval_brdf(base_color, metallic, roughness, normal, light_dir, view_dir);
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

        vec3 pl_brdf = eval_brdf(base_color, metallic, roughness, normal, light_dir, view_dir);
        vec3 light_radiance = light_intensities[i] * light_colors[i] / (r * r);
        float NdotL = max(dot(normal, light_dir), 0.0);
        float shadow = shadowCalculation(i, frag_pos);
        point_light_contribution += (1.0 - shadow) * pl_brdf * NdotL * light_radiance;
    }

    // IBL
    vec3 ambient = image_based_lighting(base_color, metallic, roughness, normal, view_dir);

    // Emission
    vec3 emission = emissive * texture(texture_emissive, frag_uv).rgb * emission_scale;
    vec3 color = (1.0 - float(use_ibl)) * view_light + (float(use_ibl)) * ambient + point_light_contribution + emission;

    float frag_dist = length(camera_pos - frag_pos);
    outColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0 / 2.4)), diffuse_lookup.w * (1.0 - pow(1.01, frag_dist - 1000)));
    outEntityID = entityID;
}