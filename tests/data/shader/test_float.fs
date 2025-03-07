#version 430 core

layout(location = 0) out vec4 FragColor;

uniform float unif;

void main()
{
    FragColor = vec4(vec3(unif), 1);
}