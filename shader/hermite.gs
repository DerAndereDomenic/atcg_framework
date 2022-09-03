#version 330 core

layout (points) in;
layout (line_strip, max_vertices=2) out;

in VS_OUT 
{
	int instance;
    vec3 points[4];
}gs_in[];

vec3 hermite(float u, vec3 p0, vec3 p1, vec3 t0, vec3 t1)
{
    vec3 p;
    //<solution>
    float F1 = 2.0*u*u*u - 3.0*u*u + 1.0;
    float F2 = -2.0*u*u*u + 3.0*u*u;
    float F3 = u*u*u - 2.0*u*u + u;
    float F4 = u*u*u - u*u;

    p = F1*p0 + F2*p1 + F3*t0 + F4*t1;
    //</solution>
    return p;
}

uniform int discretization;

void main()
{
    vec3 p0 = gs_in[0].points[0];
    vec3 t0 = gs_in[0].points[1] - p0;
    vec3 p1 = gs_in[0].points[3];
    vec3 t1 = p1 - gs_in[0].points[2];

    gl_Position = vec4(hermite(gs_in[0].instance/float(discretization), p0, p1, t0, t1),1);
    EmitVertex();
    gl_Position = vec4(hermite((gs_in[0].instance+1)/float(discretization), p0, p1, t0, t1),1);
    EmitVertex();
    EndPrimitive();
    
}