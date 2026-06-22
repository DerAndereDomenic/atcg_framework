#version 400 core

layout(location = 0) in vec3 aPosition;

out vec2 frag_uv;

void main()
{
    gl_Position = vec4(aPosition, 1);
    frag_uv = aPosition.xy * 0.5 + 0.5;
}