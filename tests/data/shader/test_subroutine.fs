#version 430 core

layout(location = 0) out vec4 FragColor;

subroutine vec3 color_fs();

subroutine(color_fs) vec3 red()
{
    return vec3(1, 0, 0);
}

subroutine(color_fs) vec3 green()
{
    return vec3(0, 1, 0);
}

subroutine(color_fs) vec3 blue()
{
    return vec3(0, 0, 1);
}

subroutine(color_fs) vec3 white()
{
    return vec3(1, 1, 1);
}

subroutine uniform color_fs get_color;

in vec3 f_color;

void main()
{
    FragColor = vec4(f_color * get_color(), 1);
}