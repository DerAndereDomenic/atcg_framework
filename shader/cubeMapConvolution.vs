#version 430 core

layout (location = 0) in vec3 aPosition;

uniform mat4 V;
uniform mat4 P;

out vec3 localPos;

void main()
{
	localPos = aPosition;

	gl_Position =  P * V * vec4(localPos, 1.0);
}