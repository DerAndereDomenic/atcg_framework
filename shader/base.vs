#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;

uniform mat4 M, V, P;

out vec3 frag_normal;
out vec3 frag_pos;

void main()
{
    gl_Position = P * V * M * vec4(aPosition, 1);
    frag_pos = vec3(M * vec4(aPosition, 1));
    frag_normal = aNormal;
}