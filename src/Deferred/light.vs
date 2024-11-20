#version 330 core
layout (location = 0) in vec3 aPos;

out vec2 frag_uv;

void main()
{
	gl_Position = vec4(aPos, 1);

	frag_uv = vec2(aPos.x*0.5 + 0.5, aPos.y*0.5 + 0.5);
}