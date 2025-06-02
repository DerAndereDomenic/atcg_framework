#version 430 core

layout(location = 0) out vec4 FragColor;

uniform mat4 unif;

void main()
{
    FragColor = unif * vec4(1);
}