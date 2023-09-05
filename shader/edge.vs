#version 430 core

// POSITION + NORMAL + TANGENT + COLOR + UV
#define VERTEX_DIM (3 + 3 + 3 + 3 + 3)

layout (location = 0) in vec2 aEdge;
layout (location = 1) in vec3 aColor; 

out vec3 color;
out vec3 start;
out vec3 end;

layout(std430, binding = 0) buffer points_layout
{
    float points[];
};

void main()
{
    start = vec3(points[VERTEX_DIM * int(aEdge.x) + 0], 
                 points[VERTEX_DIM * int(aEdge.x) + 1], 
                 points[VERTEX_DIM * int(aEdge.x) + 2]);

    end = vec3(points[VERTEX_DIM * int(aEdge.y) + 0],
               points[VERTEX_DIM * int(aEdge.y) + 1], 
               points[VERTEX_DIM * int(aEdge.y) + 2]);

    color = aColor;
}