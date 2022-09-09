#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 2) in vec3 aColor; 

out vec3 color;

uniform mat4 M;
uniform mat4 V;

void main()
{
    gl_Position = V * M * vec4(aPosition, 1.0f);
    color = aColor;
}