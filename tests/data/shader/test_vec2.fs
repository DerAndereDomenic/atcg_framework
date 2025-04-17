#version 430 core

layout(location = 0) out vec4 FragColor;

uniform vec2 unif;

void main()
{
    FragColor = vec4(unif, 0, 1);
}