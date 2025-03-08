#version 430 core

layout(location = 0) out vec4 FragColor;

uniform int unif;

void main()
{
    FragColor = vec4(vec3(unif / 255.0), 1);
}