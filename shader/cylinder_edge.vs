#version 430 core

// POSITION + NORMAL + COLOR
#define VERTEX_DIM (3 + 3 + 3)

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

// Instance variables
layout (location = 3) in vec2 aEdge;
layout (location = 4) in vec3 aEdgeColor;
layout (location = 5) in float aRadius;

uniform mat4 M, V, P;
uniform int instanced;

out vec3 frag_normal;
out vec3 frag_pos;
out vec3 frag_color;

layout(std430, binding = 0) buffer points_layout
{
    float points[];
};

void main()
{
    vec3 aInstanceStart = vec3(points[VERTEX_DIM * int(aEdge.x) + 0], 
                               points[VERTEX_DIM * int(aEdge.x) + 1], 
                               points[VERTEX_DIM * int(aEdge.x) + 2]);

    vec3 aInstanceEnd = vec3(points[VERTEX_DIM * int(aEdge.y) + 0],
                             points[VERTEX_DIM * int(aEdge.y) + 1], 
                             points[VERTEX_DIM * int(aEdge.y) + 2]);

    vec3 axis = (aInstanceEnd - aInstanceStart);
    vec3 middle_point = aInstanceStart + axis/2.0;

    mat4 model_scale = mat4(aRadius);
    model_scale[1].y = length(axis)/2.0;
    model_scale[3].w = 1;

    mat4 model_translate = mat4(1);
    model_translate[3] = vec4(middle_point, 1);

    axis = normalize(axis);
    vec3 x = normalize(cross(vec3(0, axis.z, 1.0-axis.z), axis));
    vec3 z = normalize(cross(x,axis));

    mat4 model_rotation = mat4(vec4(x,0), vec4(axis,0), vec4(z,0), vec4(0,0,0,1));

    mat4 model_edge = model_translate * model_rotation * model_scale;

    gl_Position = P * V * (M * model_edge * vec4(aPosition, 1));// + vec4(instanced * aInstanceStart, 0));
    frag_pos = vec3(M * model_edge * vec4(aPosition, 1));
    frag_normal = normalize(vec3(inverse(transpose(M * model_edge)) * vec4(aNormal, 0)));
    frag_color = aColor * aEdgeColor;
}