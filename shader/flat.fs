#version 330 core

layout (location = 0) out vec4 fragColor;

uniform vec3 flat_color;

void main()
{
    fragColor = vec4(flat_color, 1);
}