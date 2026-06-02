#version 330 core

layout(location = 0) out vec4 FragColor;

#include "common/defines.glsl"

in vec2 frag_uv;

uniform sampler2D screen_texture;
uniform isampler2D stencil_texture;
uniform float exposure;

void main()
{
    vec3 color = texture(screen_texture, frag_uv).rgb;
    uint stencil = uint(texture(stencil_texture, frag_uv).r);

    vec3 tonemapped = color;

    if(bool(stencil & uint(TONE_MAP_BIT)))
    {
        tonemapped = pow(vec3(1) - exp(-color * exposure), vec3(1.0 / 2.4));
    }

    FragColor = vec4(tonemapped, 1);
}