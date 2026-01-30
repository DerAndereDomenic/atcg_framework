#version 330 core

#include "common/defines.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out int outEntityID;
layout(location = 2) out uint outStencil;

in vec2 frag_uv;

uniform sampler2D screen_texture;
uniform int entityID;

void main()
{
    vec3 color = texture(screen_texture, frag_uv).rgb;
    FragColor = vec4(color, 1);
    outEntityID = entityID;
    outStencil = uint(0);
}