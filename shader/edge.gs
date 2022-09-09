#version 330 core

layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in vec3 color[];
out vec3 frag_color;

uniform mat4 P;

void main()
{
    gl_Position = P * gl_in[0].gl_Position;
    frag_color = color[0];
    EmitVertex();
    gl_Position = P * gl_in[1].gl_Position;
    frag_color = color[1];
    EmitVertex();

    gl_Position = P * gl_in[1].gl_Position;
    frag_color = color[1];
    EmitVertex();
    gl_Position = P * gl_in[2].gl_Position;
    frag_color = color[2];
    EmitVertex();

    gl_Position = P * gl_in[2].gl_Position;
    frag_color = color[2];
    EmitVertex();
    gl_Position = P * gl_in[0].gl_Position;
    frag_color = color[0];
    EmitVertex();

    EndPrimitive();
}