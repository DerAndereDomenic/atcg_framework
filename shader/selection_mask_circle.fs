#version 330 core

// Selection mask variant of circle.fs (used with circle.vs for PointLightComponents).
// Writes the entity id only where circle.fs would draw the circle.

layout(location = 0) out int outEntityID;

in vec3 frag_pos;

uniform float thickness;
uniform int entityID;

void main()
{
    float fade = 0.005;

    float distance = 1.0 - length(frag_pos);
    float circle = smoothstep(0.0, fade, distance);
    circle *= smoothstep(fade + thickness, thickness, distance);

    if(circle == 0.0)
        discard;

    outEntityID = entityID;
}
