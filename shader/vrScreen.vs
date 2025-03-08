#version 330 core
layout(location = 0) in vec3 aPos;

out vec2 frag_tex;

void main()
{
	gl_Position = vec4(aPos, 1);

	frag_tex = vec2(aPos.x + 1.0, aPos.y * 0.5 + 0.5);
}