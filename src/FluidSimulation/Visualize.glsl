#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) uniform image2D output_img;

uniform sampler2D veloctiy;
uniform sampler2D pressure;
uniform sampler2D dye;

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec2 velocity = texture(veloctiy, vec2(pixel_coords) / vec2(imageSize(output_img))).xy;
    float pressure = texture(pressure, vec2(pixel_coords) / vec2(imageSize(output_img))).r;

    vec3 color = texture(dye, vec2(pixel_coords) / vec2(imageSize(output_img))).rgb;

    imageStore(output_img, pixel_coords, vec4(color, 1.0));
}