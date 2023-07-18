#version 330 core

layout (location = 0) out vec4 fragColor;
layout (location = 1) out int outEntityID;

in vec3 frag_color;
in vec3 frag_pos;

uniform vec3 flat_color;
uniform vec3 camera_pos;
uniform int entityID;

void main()
{
    float frag_dist = length(camera_pos - frag_pos);
    fragColor = vec4(frag_color * flat_color, 1.0 - pow(1.01, frag_dist - 1000));
    outEntityID = entityID;
}