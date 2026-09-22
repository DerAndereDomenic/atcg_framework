#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(r32f, binding = 0) uniform image2D divergence_output;

uniform sampler2D velocity_texture;

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(divergence_output);

    vec2 uv = (vec2(pixel_coords) + vec2(0.5)) / vec2(image_size);

    int width = image_size.x;
    int height = image_size.y;

    float h = 2.0 / float(width);

    float left, right, bottom, top;

    if(pixel_coords.x == 0)
    {
        left = 0.0;
    }
    else
    {
        left = texelFetch(velocity_texture, pixel_coords - ivec2(1, 0), 0).x;
    }

    if(pixel_coords.x == width - 1)
    {
        right = 0.0;
    }
    else
    {
        right = texelFetch(velocity_texture, pixel_coords + ivec2(1, 0), 0).x;
    }

    if(pixel_coords.y == 0)
    {
        bottom = 0.0;
    }
    else
    {
        bottom = texelFetch(velocity_texture, pixel_coords - ivec2(0, 1), 0).y;
    }

    if(pixel_coords.y == height - 1)
    {
        top = 0.0;
    }
    else
    {
        top = texelFetch(velocity_texture, pixel_coords + ivec2(0, 1), 0).y;
    }

    float divergence = (right - left) / (2.0 * h) + (top - bottom) / (2.0 * h);

    imageStore(divergence_output, pixel_coords, vec4(divergence, 0.0, 0.0, 0.0));
}