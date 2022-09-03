#version 330 core

out VS_OUT
{
    out int instance;
    out vec3 points[4];
}vs_out;

uniform vec3 points[4];

void main()
{
    gl_Position = vec4(0,0,0,1);
    vs_out.instance = gl_InstanceID;
    vs_out.points[0] = points[0];
    vs_out.points[1] = points[1];
    vs_out.points[2] = points[2];
    vs_out.points[3] = points[3];
}