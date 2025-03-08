#version 430 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 g_color[];
out vec3 f_color;

subroutine vec3 multiply(vec3 color);

subroutine(multiply) vec3 multiply_half(vec3 color)
{
    return 0.5 * color;
}

subroutine(multiply) vec3 multiply_quarter(vec3 color)
{
    return 0.25 * color;
}

subroutine uniform multiply multiply_fn;

subroutine vec3 color_gs();

subroutine(color_gs) vec3 red()
{
    return vec3(1, 0, 0);
}

subroutine(color_gs) vec3 green()
{
    return vec3(0, 1, 0);
}

subroutine(color_gs) vec3 blue()
{
    return vec3(0, 0, 1);
}

subroutine uniform color_gs get_color;

void main() {
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;  // Pass through unchanged
        f_color = multiply_fn(g_color[i]) + get_color();
        EmitVertex();
    }
    EndPrimitive();
}