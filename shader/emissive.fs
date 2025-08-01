#version 330 core

layout(location = 0) out vec4 fragColor;
layout(location = 1) out int outEntityID;

in vec3 frag_pos;
in vec2 frag_uv;

uniform sampler2D texture_emissive;
uniform float emissive_scaling;
uniform vec3 camera_pos;
uniform int entityID;

void main()
{
    vec3 color_emissive = texture(texture_emissive, frag_uv).rgb;
    float frag_dist = length(camera_pos - frag_pos);
    fragColor = vec4(color_emissive * emissive_scaling, 1.0 - pow(1.01, frag_dist - 1000));
    outEntityID = entityID;
}