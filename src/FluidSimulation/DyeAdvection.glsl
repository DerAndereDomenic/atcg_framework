#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(rg32f, binding = 0) uniform image2D dye_output_texture;

uniform sampler2D prev_dye_texture;
uniform sampler2D velocity_texture;
uniform float delta_time;

int width, height;

vec2 read_velocity(vec2 position)
{
    vec2 uv = (position + 1.0) * 0.5; // Convert from [-1, 1] to [0, 1]
    return texture(velocity_texture, uv).xy;
}

vec3 read_dye(vec2 position)
{
    vec2 uv = (position + 1.0) * 0.5; // Convert from [-1, 1] to [0, 1]
    return texture(prev_dye_texture, uv).xyz;
}

vec3 advect_dye(vec2 position, vec2 velocity, float dt)
{
    vec2 prev_pos = position - velocity * dt;

    return read_dye(prev_pos);
}

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(dye_output_texture);

    width = image_size.x;
    height = image_size.y;

    vec2 uv = (vec2(pixel_coords) + vec2(0.5)) / vec2(image_size);

    vec2 position = uv * 2.0 - 1.0;

    vec2 current_velocity = read_velocity(position);

    vec3 prev_dye = advect_dye(position, current_velocity, delta_time);

    imageStore(dye_output_texture, pixel_coords, vec4(prev_dye, 0.0));
}