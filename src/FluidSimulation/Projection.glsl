#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(r32f, binding = 0) uniform image2D velocity_output;

uniform sampler2D advection_texture;
uniform sampler2D pressure_texture;
uniform float rho;
uniform float delta_time;

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(velocity_output);

    vec2 uv = (vec2(pixel_coords) + vec2(0.5)) / vec2(image_size);

    vec2 velocity = texture(advection_texture, uv).xy;

    float h = 2.0 / float(image_size.x);

    ivec2 left_coords = clamp(pixel_coords + ivec2(-1, 0), ivec2(0, 0), image_size - ivec2(1, 1));
    ivec2 right_coords = clamp(pixel_coords + ivec2(1, 0), ivec2(0, 0), image_size - ivec2(1, 1));
    ivec2 bottom_coords = clamp(pixel_coords + ivec2(0, -1), ivec2(0, 0), image_size - ivec2(1, 1));
    ivec2 top_coords = clamp(pixel_coords + ivec2(0, 1), ivec2(0, 0), image_size - ivec2(1, 1));

    float left_pressure = texelFetch(pressure_texture, left_coords, 0).r;
    float right_pressure = texelFetch(pressure_texture, right_coords, 0).r;
    float bottom_pressure = texelFetch(pressure_texture, bottom_coords, 0).r;
    float top_pressure = texelFetch(pressure_texture, top_coords, 0).r;

    float pressure_gradient_x = (right_pressure - left_pressure) / (2.0 * h);
    float pressure_gradient_y = (top_pressure - bottom_pressure) / (2.0 * h);

    velocity -= (delta_time / rho) * vec2(pressure_gradient_x, pressure_gradient_y);

    imageStore(velocity_output, pixel_coords, vec4(velocity, 0.0, 0.0));
}