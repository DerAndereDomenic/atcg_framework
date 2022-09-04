#version 330 core

layout (location = 0) out vec4 outColor;

uniform vec3 flat_color;

void main()
{
    outColor = vec4(flat_color, 1);
}