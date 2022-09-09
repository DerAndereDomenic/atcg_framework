#version 330 core

layout (location = 0) out vec4 outColor;

in vec3 frag_color;

uniform vec3 flat_color;

void main()
{
    outColor = vec4(flat_color * frag_color, 1);
}