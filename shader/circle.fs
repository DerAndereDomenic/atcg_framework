#version 330 core

layout (location = 0) out vec4 outColor;

in vec3 frag_pos;

uniform vec3 flat_color;
uniform float thickness;

void main()
{
    float fade = 0.005f;

    float distance = 1.0f - length(frag_pos);
    float circle = smoothstep(0.0f, fade, distance);
    circle *= smoothstep(fade + thickness, thickness, distance);

    if(circle == 0.0)
        discard;

    outColor = vec4(flat_color, circle);
}