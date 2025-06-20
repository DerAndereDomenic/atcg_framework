#version 430 core

layout(location = 0) in vec3 aPosition;

subroutine vec3 color();

subroutine(color) vec3 red()
{
    return vec3(1, 0, 0);
}

subroutine(color) vec3 green()
{
    return vec3(0, 1, 0);
}

subroutine(color) vec3 blue()
{
    return vec3(0, 0, 1);
}

subroutine uniform color get_color;

out vec3 g_color;

void main()
{
    g_color = get_color();
    gl_Position = vec4(aPosition, 1);
}