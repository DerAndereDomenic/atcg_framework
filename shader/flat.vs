#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 2) in vec3 aColor;

out vec3 frag_color;

uniform mat4 M, V, P;

void main()
{
    gl_Position = P * V * M * vec4(aPosition, 1);
    frag_color = aColor;
}