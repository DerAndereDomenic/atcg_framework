#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;

out vec2 frag_uv;

void main()
{
    gl_Position = vec4(aPosition, 1);
    frag_uv = aUV;
}