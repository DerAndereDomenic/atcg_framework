#version 330 core

layout (points) in;
layout (line_strip, max_vertices=2) out;

in VS_OUT 
{
	int instance;
    vec3 points[4];
}gs_in[];

vec3 bezier(float u)
{
    float B0 = (1.-u)*(1.-u)*(1.-u);
    float B1 = 3.*u*(1.-u)*(1.-u);
    float B2 = 3.*u*u*(1.-u);
    float B3 = u*u*u;

    return B0*gs_in[0].points[0]+B1*gs_in[0].points[1]+B2*gs_in[0].points[2]+B3*gs_in[0].points[3];
}

uniform int discretization;

void main()
{
    gl_Position = vec4(bezier(gs_in[0].instance/float(discretization)),1);
    EmitVertex();
    gl_Position = vec4(bezier((gs_in[0].instance+1)/float(discretization)),1);
    EmitVertex();
    EndPrimitive();
    
}