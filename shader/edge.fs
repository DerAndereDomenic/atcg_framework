#version 430 core

layout (location = 0) out vec4 outColor;

in vec3 frag_color;
in vec3 frag_pos;

uniform vec3 flat_color;
uniform vec3 camera_pos;

void main()
{
    float frag_dist = length(camera_pos - frag_pos);
    outColor = vec4(flat_color * frag_color, 1.0 - pow(1.01, frag_dist - 1000));
}