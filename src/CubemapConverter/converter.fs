#version 450

layout (location = 0) out vec4 FragColor;

uniform samplerCube cubemap;

in vec2 spherical;

void main()
{
    vec3 dir = vec3(sin(spherical.y) * cos(spherical.x),
                    cos(spherical.y),
                    sin(spherical.y) * sin(spherical.x));

    FragColor = vec4(texture(cubemap, dir).rgb, 1);
}