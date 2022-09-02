#version 330 core

layout (location = 0) out vec4 outColor;

in vec3 frag_normal;

uniform vec3 camera_pos;

void main()
{
    outColor = vec4(frag_normal,1);
}