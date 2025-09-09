#version 430 core

#include "common/defines.glsl"

layout(location = 0) out vec4 FragColor;
layout(location = 1) out int entityID;
layout(location = 2) out uint outStencil;

in vec3 frag_tex;

uniform samplerCube skybox;

void main()
{
    vec3 color = texture(skybox, frag_tex).rgb;
    FragColor = vec4(color, 1.0);
    // FragColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0 / 2.4)), 1.0);
    entityID = -1;
    outStencil = uint(TONE_MAP_BIT);
}