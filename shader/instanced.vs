#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aUV;

// Instance variables
layout (location = 5) in mat4 aInstanceModel;
layout (location = 9) in vec3 aInstanceColor;

uniform mat4 M, V, P;

out vec3 frag_normal;
out vec3 frag_pos;
out vec3 frag_color;
out vec2 frag_uv;
out mat3 frag_tbn;

void main()
{
    frag_pos = vec3(M * aInstanceModel * vec4(aPosition, 1.0));

    gl_Position = P * V * vec4(frag_pos, 1);

    // This could eventually lead to problems if we allow the client to do instance rendering of arbitrary meshes
    // frag_normal = normalize(vec3(inverse(transpose((1-instanced) * M + instanced * M)) * vec4(aNormal, 0)));
    mat4 normal_matrix = transpose(inverse(M * aInstanceModel));
    vec3 axis = normalize(vec3(normal_matrix * vec4(aNormal, 0)));
    vec3 tangent = normalize(vec3(normal_matrix * vec4(aTangent + 1e-5, 0))); // Numerical stability
    vec3 bitangent = normalize(cross(axis, tangent));
    mat3 tbn = mat3(tangent, bitangent, axis);
    frag_tbn = tbn;
    frag_normal = axis;
    frag_color = aColor * aInstanceColor;

    frag_uv = aUV.xy;
}