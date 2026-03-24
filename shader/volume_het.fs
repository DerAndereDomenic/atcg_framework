#version 330 core

#include "common/defines.glsl"

layout(location = 0) out vec4 fragColor;
layout(location = 1) out int outEntityID;
layout(location = 2) out uint outStencil;

uniform int entityID;

uniform sampler2D front_depth;
uniform sampler2D back_depth;
uniform sampler3D density_grid;
uniform float density_scale;
uniform mat4 density_to_uvw;

uniform sampler3D albedo_grid;
uniform float albedo_scale;
uniform mat4 albedo_to_uvw;

uniform sampler3D emission_grid;
uniform float emission_scale;
uniform mat4 emission_to_uvw;

uniform float g; // Anisotropy factor

uniform mat4 invView, invProj;

vec3 reconstructWorldPosition(vec2 uv, float depth)
{
    // 1. NDC
    vec4 ndc;
    ndc.xy = uv * 2.0 - 1.0;
    ndc.z  = depth * 2.0 - 1.0;
    ndc.w  = 1.0;

    // 2. Clip → View
    vec4 viewPos = invProj * ndc;
    viewPos /= viewPos.w;

    // 3. View → World
    vec4 worldPos = invView * viewPos;
    return worldPos.xyz;
}

struct RayMarchResult
{
    float transmittance;
    vec3 scattering;
};

RayMarchResult estimate_transmittance(vec3 start, vec3 end)
{
    RayMarchResult result;
    result.scattering = vec3(0.0);
    int n_samples = 64;
    float dt = length(end - start) / float(n_samples);
    vec3 dir = normalize(end - start);

    float transmittance = 1.0;
    for(int i = 0; i < n_samples; ++i)
    {
        vec3 sample_pos = start + dir * (dt * (float(i) + 0.5));

        // Density
        vec4 uvw = density_to_uvw * vec4(sample_pos, 1.0);
        float density_sample = density_scale * texture(density_grid, uvw.xyz).r;

        // Albedo
        uvw = albedo_to_uvw * vec4(sample_pos, 1.0);
        vec3 albedo_sample = albedo_scale * texture(albedo_grid, uvw.xyz).rgb;

        // Emission
        uvw = emission_to_uvw * vec4(sample_pos, 1.0);
        vec3 emission_sample = emission_scale * texture(emission_grid, uvw.xyz).rgb;

        float delta_transmittance = exp(-density_sample * dt);

        result.scattering += (albedo_sample * density_sample + emission_sample) * transmittance * dt;
        transmittance *= delta_transmittance;

    }
    result.transmittance = transmittance;
    return result;
}

void main()
{
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(front_depth, 0));
    float front_depth_value = texture(front_depth, uv).r;
    float back_depth_value = texture(back_depth, uv).r;

    vec3 worldPosFront = reconstructWorldPosition(uv, front_depth_value);
    vec3 worldPosBack = reconstructWorldPosition(uv, back_depth_value);

    RayMarchResult marchResult = estimate_transmittance(worldPosFront, worldPosBack);

    fragColor = vec4(marchResult.scattering , 1.0 - marchResult.transmittance);
    outEntityID = entityID;
    outStencil = uint(TONE_MAP_BIT);
}