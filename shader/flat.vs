#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 2) in vec3 aColor;

// Instance variables
layout (location = 3) in vec3 aInstanceOffset;
layout (location = 5) in vec3 aInstanceColor;

out vec3 frag_color;
out vec3 frag_pos;

uniform mat4 M, V, P;
uniform float point_size;
uniform int instanced;

void main()
{
    vec3 scale_model = vec3(length(M[0]), length(M[1]), length(M[2]));
    vec3 scale_point = instanced * vec3(point_size) + (1 - instanced) * scale_model;

    mat4 inv_scale_model = mat4(1);
    inv_scale_model[0][0] = 1.0/scale_model.x;
    inv_scale_model[1][1] = 1.0/scale_model.y;
    inv_scale_model[2][2] = 1.0/scale_model.z;

    mat4 scale_primitive = mat4(1);
    scale_primitive[0][0] = scale_point.x;
    scale_primitive[1][1] = scale_point.y;
    scale_primitive[2][2] = scale_point.z;

    frag_pos = vec3(M * inv_scale_model * scale_primitive * vec4(aPosition, 1) + instanced * M * vec4(aInstanceOffset, 0));

    gl_Position = P * V * vec4(frag_pos, 1);

    frag_color = aColor * (instanced * aInstanceColor + (1 - instanced) * vec3(1));
}