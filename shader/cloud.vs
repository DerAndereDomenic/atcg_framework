#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

out vec3 frag_pos_model;
flat out vec3 camera_pos_model;

uniform vec3 camera_pos;
uniform mat4 M, V, P;

void main()
{
    mat4 invM = inverse(M);
    camera_pos_model = vec3(invM*vec4(camera_pos,1));
    frag_pos_model = aPosition;
    gl_Position = P * V * M * vec4(aPosition, 1);
}