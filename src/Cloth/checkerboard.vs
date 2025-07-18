#version 430 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aUV;

// Instance variables
layout (location = 5) in vec3 aInstanceOffset;
layout (location = 6) in vec3 aInstanceColor;

uniform mat4 M, V, P;
uniform int instanced;

out vec3 frag_normal;
out vec3 frag_pos;
out vec3 frag_color;
out vec2 frag_uv;

void main()
{
    gl_Position = P * V * (M * vec4(aPosition, 1) + vec4(instanced * aInstanceOffset, 0));
    frag_pos = vec3(M * vec4(aPosition, 1));
    frag_normal = normalize(vec3(inverse(transpose(M)) * vec4(aNormal, 0)));
    frag_color = aColor * (instanced * aInstanceColor + (1 - instanced) * vec3(1));

    frag_uv = aPosition.xz/2.0 + 0.5;
}