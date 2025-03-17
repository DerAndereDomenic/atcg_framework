#version 430 core
layout(location = 0) in vec3 aPosition;

out vec3 localPos;

uniform mat4 P;
uniform mat4 V;

void main()
{
    localPos = aPosition;
    gl_Position = P * V * vec4(localPos, 1.0);
}