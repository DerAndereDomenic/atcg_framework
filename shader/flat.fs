#version 330 core

layout (location = 0) out vec4 fragColor;

in vec3 frag_color;

uniform vec3 flat_color;

void main()
{
    fragColor = vec4(frag_color * flat_color, 1);
}