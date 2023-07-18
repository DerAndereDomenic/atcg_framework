#version 430 core

layout (points) in;
layout (line_strip, max_vertices = 2) out;

in vec3 color[];
in vec3 start[];
in vec3 end[];
out vec3 frag_color;
out vec3 frag_pos;

uniform mat4 M,V,P;

void main()
{
    vec3 frag_pos = vec3(M * vec4(start[0], 1));
    gl_Position = P * V * vec4(frag_pos, 1);
    frag_color = color[0];
    EmitVertex();

    frag_pos = vec3(M * vec4(end[0], 1));
    gl_Position = P * V * vec4(frag_pos,1);
    frag_color = color[0];
    EmitVertex();

    EndPrimitive();
}