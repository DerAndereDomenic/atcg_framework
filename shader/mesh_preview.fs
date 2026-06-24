#version 330 core

layout(location = 0) out vec4 outColor;

in vec3 frag_normal;

uniform vec3 camera_dir;

void main()
{
    float NdotL = max(dot(frag_normal, camera_dir), 0.0);

    vec3 color = vec3(NdotL);

    color = pow(vec3(1) - exp(-color), vec3(1.0 / 2.4));

    outColor = vec4(color, 1.0);
}