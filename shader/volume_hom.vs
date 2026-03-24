#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec3 aTangent;
layout(location = 4) in vec3 aUV;

uniform mat4 M, V, P;

void main()
{
    gl_Position = P * V * M * vec4(aPosition, 1);
}