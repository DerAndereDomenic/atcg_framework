#version 430 core

layout (location = 0) out vec4 outColor;
layout (location = 1) out int outEntityID;

in vec3 frag_color;
in vec3 frag_pos;

uniform vec3 flat_color;
uniform vec3 camera_pos;
uniform float base_transparency;
uniform float fall_off_edge;
uniform int entityID;

void main()
{
    float frag_dist = length(camera_pos.xz - frag_pos.xz);
    float d = fall_off_edge * 7.0 + 10.0;
    float fall_off = 1.0 - smoothstep(d*0.3, d*1.1, frag_dist);
    outColor = vec4(flat_color * frag_color, base_transparency * fall_off);
    outEntityID = entityID;
}