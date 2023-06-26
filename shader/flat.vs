#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 2) in vec3 aColor;

// Instance variables
layout (location = 3) in vec3 aInstanceOffset;
layout (location = 5) in vec3 aInstanceColor;

out vec3 frag_color;

uniform mat4 M, V, P;
uniform int instanced;

void main()
{
    //TODO: Fix scaling of point size
    gl_Position = P * V * (M * vec4(aPosition, 1) + vec4(aInstanceOffset * instanced, 0));
    frag_color = aColor * (instanced * aInstanceColor + (1 - instanced) * vec3(1));;
}