#version 330 core

layout(location = 0) out vec4 outColor;
layout(location = 1) out int outEntityID;

in vec3 frag_pos;

uniform vec3 flat_color;
uniform float thickness;
uniform int entityID = -1;

void main()
{
    float fade = 0.005;

    float distance = 1.0 - length(frag_pos);
    float circle = smoothstep(0.0, fade, distance);
    circle *= smoothstep(fade + thickness, thickness, distance);

    if(circle == 0.0)
        discard;

    outColor = vec4(flat_color, circle);
    outEntityID = entityID;
}