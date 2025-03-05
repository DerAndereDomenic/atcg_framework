#version 330 core

layout(location = 0) in vec3 aPosition;

uniform mat4 V, P;
uniform vec3 position;
uniform float radius;

out vec3 frag_pos;

void main()
{
    vec3 cam_right = vec3(transpose(V)[0]);
    vec3 cam_up = vec3(transpose(V)[1]);

    gl_Position = P * V * vec4(position + cam_right * aPosition.x * radius + cam_up * aPosition.y * radius, 1);
    frag_pos = aPosition;
}