#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 2) in vec3 aNormal;

uniform mat4 M, V, P;
uniform vec3 camera_pos;

out vec3 frag_normal;

void main()
{
    frag_normal = normalize(vec3(inverse(transpose(M)) * vec4(aNormal, 0)));

    if(dot(frag_normal, normalize(camera_pos - vec3(M * vec4(aPosition, 1)))) < 0)
    {
        frag_normal *= -1.0;
    }

    gl_Position = P * V * M * vec4(aPosition, 1);
}