#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(rg32f, binding = 0) uniform image2D dye_output_texture;

uniform sampler2D prev_dye_texture;
uniform float delta_time;

int width, height;
float D = 3e-5;

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(dye_output_texture);

    width = image_size.x;
    height = image_size.y;

    float h = 2.0 / float(width);

    vec2 left_coord = clamp(pixel_coords - ivec2(1, 0), ivec2(0, 0), image_size - ivec2(1, 1));
    vec2 right_coord = clamp(pixel_coords + ivec2(1, 0), ivec2(0, 0), image_size - ivec2(1, 1));
    vec2 bottom_coord = clamp(pixel_coords - ivec2(0, 1), ivec2(0, 0), image_size - ivec2(1, 1));
    vec2 top_coord = clamp(pixel_coords + ivec2(0, 1), ivec2(0, 0), image_size - ivec2(1, 1));

    vec3 left_dye = texelFetch(prev_dye_texture, ivec2(left_coord), 0).xyz;
    vec3 right_dye = texelFetch(prev_dye_texture, ivec2(right_coord), 0).xyz;
    vec3 bottom_dye = texelFetch(prev_dye_texture, ivec2(bottom_coord), 0).xyz;
    vec3 top_dye = texelFetch(prev_dye_texture, ivec2(top_coord), 0).xyz;

    vec3 current_dye = texelFetch(prev_dye_texture, pixel_coords, 0).xyz;

    vec3 new_dye = current_dye + D * delta_time / (h * h) * (left_dye + right_dye + bottom_dye + top_dye - 4.0 * current_dye);

    imageStore(dye_output_texture, pixel_coords, vec4(new_dye, 0.0));
}