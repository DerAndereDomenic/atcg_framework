#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(r32f, binding = 0) uniform image2D pressure_output;

uniform sampler2D prev_pressure_texture;
uniform sampler2D divergence_texture;
uniform float delta_time;
uniform float rho;

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(pressure_output);

    vec2 uv = (vec2(pixel_coords) + vec2(0.5)) / vec2(image_size);

    int width = image_size.x;
    int height = image_size.y;

    ivec2 left_coords = clamp(pixel_coords - ivec2(1, 0), ivec2(0, 0), image_size - ivec2(1, 1));
    ivec2 right_coords = clamp(pixel_coords + ivec2(1, 0), ivec2(0, 0), image_size - ivec2(1, 1));
    ivec2 bottom_coords = clamp(pixel_coords - ivec2(0, 1), ivec2(0, 0), image_size - ivec2(1, 1));
    ivec2 top_coords = clamp(pixel_coords + ivec2(0, 1), ivec2(0, 0), image_size - ivec2(1, 1));

    float left = texelFetch(prev_pressure_texture, left_coords, 0).r;
    float right = texelFetch(prev_pressure_texture, right_coords, 0).r;
    float bottom = texelFetch(prev_pressure_texture, bottom_coords, 0).r;
    float top = texelFetch(prev_pressure_texture, top_coords, 0).r;

    float h = 2.0 / float(width); // Assuming square cells, h is the cell size

    float divergence = texelFetch(divergence_texture, pixel_coords, 0).r;

    float rhs = -divergence * rho / delta_time;

    float pressure = (left + right + bottom + top + h * h * rhs) / 4.0;
    imageStore(pressure_output, pixel_coords, vec4(pressure, 0.0, 0.0, 0.0));
}