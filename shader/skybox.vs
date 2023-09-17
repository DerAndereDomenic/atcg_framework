#version 430 core
layout (location = 0) in vec3 aPosition;

out vec3 frag_tex;

uniform mat4 P;
uniform mat4 V;

void main()
{
    frag_tex = aPosition;
    gl_Position = P * V * vec4(aPosition, 1.0);
} 