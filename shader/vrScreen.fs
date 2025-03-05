#version 330 core

out vec4 FragColor;

in vec2 frag_tex;

uniform sampler2D texture_left;
uniform sampler2D texture_right;

void main()
{
	float u = frag_tex.x;
	float v = frag_tex.y;
	if(u <= 1.0)
	{
		FragColor = texture(texture_left, frag_tex);
	}
	else
	{
		FragColor = texture(texture_right, vec2(u - 1, v));
	}
}