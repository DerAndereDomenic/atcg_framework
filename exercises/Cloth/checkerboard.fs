#version 430 core

layout (location = 0) out vec4 outColor;
layout (location = 1) out int outEntityID;

in vec3 frag_normal;
in vec3 frag_pos;
in vec3 frag_color;
in vec2 frag_uv;

uniform vec3 camera_pos;
uniform vec3 camera_dir;
uniform vec3 flat_color;
uniform float checker_size;
uniform int entityID;

void main()
{
    vec2 checker_pattern = floor(mod(frag_uv, vec2(2.0 * checker_size)) * 1.0/checker_size);
    float white = checker_pattern.x == checker_pattern.y ? 1.0 : 0.0;
    vec3 color_combined = white * frag_color * flat_color;
    vec3 ambient = vec3(0.1f) * color_combined;
    vec3 light_radiance = vec3(2.0);
    float light_distance = 1.0f;
    vec3 light_dir = normalize(camera_dir);

    vec3 color = color_combined / 3.14159f * light_radiance * max(0.0f, dot(frag_normal, light_dir)) + ambient;

    outColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0/2.4)),1);
    outEntityID = entityID;
}