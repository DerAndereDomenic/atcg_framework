#version 330 core

out VS_OUT
{
    out int instance;
    out vec3 points[4];
}vs_out;

uniform vec3 points[4];

uniform mat4 V;

void main()
{
    gl_Position = vec4(0,0,0,1);
    vs_out.instance = gl_InstanceID;
    vs_out.points[0] = vec3(V * vec4(points[0],1));
    vs_out.points[1] = vec3(V * vec4(points[1],1));
    vs_out.points[2] = vec3(V * vec4(points[2],1));
    vs_out.points[3] = vec3(V * vec4(points[3],1));
}