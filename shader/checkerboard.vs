#version 400 core

layout(location = 0) in vec3 aPosition;
layout(location = 2) in vec3 aNormal;
layout(location = 4) in vec3 aUV;

uniform mat4 M, V, P;
uniform vec3 camera_pos;

out vec3 frag_normal;
out vec2 frag_uv;
out vec3 frag_pos;

void main()
{
    frag_pos = vec3(M * vec4(aPosition, 1.0));

    gl_Position = P * V * vec4(frag_pos, 1);

    // Calculate tangent vectors
    mat4 normal_matrix = transpose(inverse(M)); //TODO: Compute on host

    frag_normal = normalize(vec3(normal_matrix * vec4(aNormal, 0)));

    if(dot(frag_normal, normalize(camera_pos - frag_pos)) < 0)
    {
        frag_normal *= -1.0;
    }

    frag_uv = aUV.xy;
}