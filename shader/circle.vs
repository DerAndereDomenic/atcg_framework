#version 330 core

layout (location = 0) in vec3 aPosition;

uniform mat4 M,V,P;

out vec3 frag_pos;

void main()
{
    gl_Position = P * V * M * vec4(aPosition, 1);
    frag_pos = aPosition;
}