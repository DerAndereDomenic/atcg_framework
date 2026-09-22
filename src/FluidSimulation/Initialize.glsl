#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(rg32f, binding = 0) uniform image2D velocity_output;
layout(r32f, binding = 1) uniform image2D pressure_output;
layout(rgba32f, binding = 2) uniform image2D dye_output;

vec2 spawn_vortex(vec2 current_position, vec2 vortex_position, float amplitude, float sigma_sq)
{
    float r_sq = dot(current_position - vortex_position, current_position - vortex_position);
    vec2 velocity = vec2(-current_position.y, current_position.x) * amplitude * exp(-r_sq / sigma_sq);

    return velocity;
}

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(velocity_output);

    vec2 uv = (vec2(pixel_coords) + vec2(0.5)) / vec2(image_size);

    vec2 position = uv * 2.0 - 1.0;

    vec2 velocity = vec2(0.0, 0.0);
    velocity += spawn_vortex(position, vec2(-0.5, -0.5), 2.0, 0.4 * 0.4);
    velocity += spawn_vortex(position, vec2(0.5, 0.5), 2.0, 0.4 * 0.4);

    float pressure = 0.0;

    vec2 center_red = vec2(-0.5, -0.3);
    vec2 center_blue = vec2(0.5, 0.3);
    vec2 center_green = vec2(0.1, -0.2);

    vec3 dye = vec3(0.0, 0.0, 0.0); // Default to black
    if(dot(position - center_red, position - center_red) < 0.05)
    {
        dye = vec3(1.0, 0.0, 0.0); // Red
    }
    if(dot(position - center_blue, position - center_blue) < 0.05)
    {
        dye = vec3(0.0, 0.0, 1.0); // Blue
    }
    if(dot(position - center_green, position - center_green) < 0.05)
    {
        dye = vec3(0.0, 1.0, 0.0); // Green
    }

    imageStore(velocity_output, pixel_coords, vec4(velocity, 0.0, 0.0));
    imageStore(pressure_output, pixel_coords, vec4(pressure, 0.0, 0.0, 0.0));
    imageStore(dye_output, pixel_coords, vec4(dye, 1.0));
}