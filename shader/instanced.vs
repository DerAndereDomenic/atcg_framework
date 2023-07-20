#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

// Instance variables
layout (location = 3) in mat4 aInstanceModel;
layout (location = 7) in vec3 aInstanceColor;

uniform mat4 M, V, P;

out vec3 frag_normal;
out vec3 frag_pos;
out vec3 frag_color;

void main()
{
    frag_pos = vec3(M * aInstanceModel * vec4(aPosition, 1.0));

    gl_Position = P * V * vec4(frag_pos, 1);

    // This could eventually lead to problems if we allow the client to do instance rendering of arbitrary meshes
    // frag_normal = normalize(vec3(inverse(transpose((1-instanced) * M + instanced * M)) * vec4(aNormal, 0)));
    frag_normal = normalize(vec3(M * aInstanceModel * vec4(aNormal, 0)));
    frag_color = aColor * aInstanceColor;
}