#version 330 core

layout (location = 0) out vec4 outColor;

in vec3 frag_normal;

void main()
{
    outColor = vec4(frag_normal,1);
}