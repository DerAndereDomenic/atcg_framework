#version 330 core

layout (location = 0) out vec4 fragColor;
layout (location = 1) out int outEntityID;

in vec3 frag_color;

uniform vec3 flat_color;
uniform int entityID;

void main()
{
    fragColor = vec4(frag_color * flat_color, 1);
    outEntityID = entityID;
}