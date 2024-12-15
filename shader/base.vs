#version 400 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aUV;

// Instance variables
layout (location = 5) in vec3 aInstanceOffset;
layout (location = 6) in vec3 aInstanceColor;

uniform mat4 M, V, P;
uniform float point_size;
uniform int instanced;

out vec3 frag_normal;
out vec3 frag_pos;
out vec3 frag_color;
out vec2 frag_uv;
out mat3 frag_tbn;

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

    // This could eventually lead to problems if we allow the client to do instance rendering of arbitrary meshes
    // frag_normal = normalize(vec3(inverse(transpose((1-instanced) * M + instanced * M)) * vec4(aNormal, 0)));

    // Calculate tangent vectors
    mat4 normal_matrix = transpose(inverse(M)); //TODO: Compute on host
    vec3 axis = normalize(vec3(normal_matrix * vec4(aNormal, 0)));
    vec3 tangent = normalize(vec3(normal_matrix * vec4(aTangent + 1e-5, 0))); // Numerical stability
    vec3 bitangent = normalize(cross(axis, tangent));
    mat3 tbn = mat3(tangent, bitangent, axis);
    frag_tbn = tbn;
    frag_normal = axis;
    frag_color = aColor * (instanced * aInstanceColor + (1 - instanced) * vec3(1));

    frag_uv = aUV.xy;
}