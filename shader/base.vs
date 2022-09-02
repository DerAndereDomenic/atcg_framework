#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aColor;

uniform mat4 M, V, P;

out vec3 frag_color;

void main()
{
    gl_Position = P * V * M * vec4(aPosition, 1);
    frag_color = aColor;
}