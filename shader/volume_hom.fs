#version 330 core

#include "common/defines.glsl"

layout(location = 0) out vec4 fragColor;
layout(location = 1) out int outEntityID;
layout(location = 2) out uint outStencil;

uniform int entityID;

uniform sampler2D back_depth;
uniform vec3 albedo;
uniform float density;
uniform float g; // Anisotropy factor
uniform float Le; // Emission strength
uniform vec3 Le_color; // Emission color

uniform mat4 invView, invProj;

vec3 reconstructWorldPosition(vec2 uv, float depth)
{
    // 1. NDC
    vec4 ndc;
    ndc.xy = uv * 2.0 - 1.0;
    ndc.z = depth * 2.0 - 1.0;
    ndc.w = 1.0;

    // 2. Clip → View
    vec4 viewPos = invProj * ndc;
    viewPos /= viewPos.w;

    // 3. View → World
    vec4 worldPos = invView * viewPos;
    return worldPos.xyz;
}

void main()
{
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(back_depth, 0));
    float front_depth_value = gl_FragCoord.z;
    float back_depth_value = texture(back_depth, uv).r;

    vec3 worldPosFront = reconstructWorldPosition(uv, front_depth_value);
    vec3 worldPosBack = reconstructWorldPosition(uv, back_depth_value);

    float dt = length(worldPosBack - worldPosFront);
    float transmittance = exp(-density * dt);

    vec3 scattering = albedo;
    vec3 emission = Le_color * Le / density;
    scattering += emission;

    fragColor = vec4(scattering, 1.0 - transmittance);
    outEntityID = entityID;
    outStencil = uint(TONE_MAP_BIT);
}