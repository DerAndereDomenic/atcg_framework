#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(rg32f, binding = 0) uniform image2D velocity_output_texture;

uniform sampler2D prev_velocity_texture;
uniform float delta_time;

int width, height;

vec2 read_velocity(vec2 position)
{
    vec2 uv = (position + 1.0) * 0.5; // Convert from [-1, 1] to [0, 1]
    return texture(prev_velocity_texture, uv).xy;
}

vec2 advect_velocity(vec2 position, vec2 velocity, float dt)
{
    vec2 prev_pos = position - velocity * dt;

    return read_velocity(prev_pos);
}

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(velocity_output_texture);

    width = image_size.x;
    height = image_size.y;

    vec2 uv = (vec2(pixel_coords) + vec2(0.5)) / vec2(image_size);

    vec2 position = uv * 2.0 - 1.0;

    vec2 current_velocity = read_velocity(position);

    // 1. Advect the velocity field using the current velocity and delta time
    vec2 prev_velocity = advect_velocity(position, current_velocity, delta_time);

    imageStore(velocity_output_texture, pixel_coords, vec4(prev_velocity, 0.0, 0.0));
}