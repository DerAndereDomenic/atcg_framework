#version 450

layout (location = 0) in vec3 aPosition;

out vec2 spherical;

void main()
{
    gl_Position = vec4(aPosition, 1);

    spherical.x = (aPosition.x + 1.0) * 3.14159;
    spherical.y = (aPosition.y + 1.0)/2.0 * 3.14159;
}