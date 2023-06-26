#version 430 core

layout (points) in;
layout (line_strip, max_vertices = 2) out;

in vec3 color[];
in vec3 start[];
in vec3 end[];
out vec3 frag_color;

uniform mat4 M,V,P;

void main()
{
    gl_Position = P * V * M * vec4(start[0],1);
    frag_color = color[0];
    EmitVertex();

    gl_Position = P * V * M * vec4(end[0],1);
    frag_color = color[0];
    EmitVertex();

    EndPrimitive();
}