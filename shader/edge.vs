#version 330 core

layout (location = 0) in vec3 aPosition;

uniform mat4 M;
uniform mat4 V;

void main()
{
    gl_Position = V * M * vec4(aPosition, 1.0f);
}