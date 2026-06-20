#version 400 core

#include "common/defines.glsl"
#include "common/bsdf.glsl"
#include "common/ibl.glsl"

layout(location = 0) out vec4 outColor;
layout(location = 1) out int outEntityID;
layout(location = 2) out uint outStencil;

in vec3 frag_normal;
in vec3 frag_pos;
in vec2 frag_uv;

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
uniform sampler2D texture_ior;

// Light data
uniform vec3 light_colors[MAX_LIGHTS];
uniform float light_intensities[MAX_LIGHTS];
uniform vec3 light_positions[MAX_LIGHTS];
uniform samplerCubeArray shadow_maps;
uniform int num_lights = 0;
uniform int receive_shadow;
uniform int shadow_pass = 0;

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

    return shadow_pass * receive_shadow * shadow;
}

vec3 checkerboard_color(vec3 pos)
{
    float scale = 70.0;
    float checker = mod(floor(pos.x * scale) + floor(pos.y * scale) + floor(pos.z * scale), 2.0);
    return mix(vec3(1), vec3(0.0), checker);
}

void main()
{
    // Set globals
    vec3 view_dir = normalize(camera_pos - frag_pos);
    vec3 diffuse_lookup = checkerboard_color(vec3(frag_uv, 0));
    vec3 base_color = diffuse_lookup;
    float roughness = 1.0;
    float metallic = 0.0;
    float ior = 1.5;
    vec3 normal = frag_normal;

    // Define colocated direction light
    vec3 light_radiance = vec3(2.0);
    vec3 light_dir = normalize(camera_dir);

    // PBR material shader
    vec3 view_light = vec3(0);
    if(num_lights <= 0)
    {
        vec3 brdf = eval_brdf(base_color, metallic, roughness, ior, normal, light_dir, view_dir);
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

        vec3 pl_brdf = eval_brdf(base_color, metallic, roughness, ior, normal, light_dir, view_dir);
        vec3 light_radiance = light_intensities[i] * light_colors[i] / (r * r);
        float NdotL = max(dot(normal, light_dir), 0.0);
        float shadow = shadowCalculation(i, frag_pos);
        point_light_contribution += (1.0 - shadow) * pl_brdf * NdotL * light_radiance;
    }

    // IBL
    vec3 ambient = image_based_lighting(base_color, metallic, roughness, ior, normal, view_dir);

    vec3 color = (1.0 - float(use_ibl)) * view_light + (float(use_ibl)) * ambient + point_light_contribution;

    float frag_dist = length(camera_pos - frag_pos);
    // outColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0 / 2.4)), diffuse_lookup.w * (1.0 - pow(1.01, frag_dist - 1000)));
    outColor = vec4(color, (1.0 - pow(1.01, frag_dist - 1000)));
    outEntityID = entityID;
    outStencil = uint(TONE_MAP_BIT);
}