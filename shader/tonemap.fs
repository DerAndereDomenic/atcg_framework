#version 330 core

layout(location = 0) out vec4 FragColor;
layout(location = 1) out int entityID;
layout(location = 2) out uint outStencil;

#include "common/defines.glsl"

in vec2 frag_uv;

uniform sampler2D screen_texture;
uniform isampler2D entity_texture;
uniform isampler2D stencil_texture;

void main()
{
    vec3 color = texture(screen_texture, frag_uv).rgb;
    int entity_id = texture(entity_texture, frag_uv).r;
    uint stencil = uint(texture(stencil_texture, frag_uv).r);

    vec3 tonemapped = color;
    
    if(bool(stencil & uint(TONE_MAP_BIT)))
    {
        tonemapped = pow(vec3(1) - exp(-color), vec3(1.0 / 2.4));
    }

    FragColor = vec4(tonemapped, 1);
    entityID = entity_id;
    outStencil = stencil;
}