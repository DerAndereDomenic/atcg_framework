#version 330 core

layout(location = 0) in vec3 aPosition;

out vec2 frag_uv;

uniform mat4 M, V, P;

void main()
{
    gl_Position = P * V * M * vec4(aPosition, 1);
    frag_uv = aPosition.xy * 0.5 + 0.5;
}