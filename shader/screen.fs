#version 330 core

out vec4 FragColor;

in vec2 frag_uv;

uniform sampler2D screen_texture;

void main()
{
    vec3 color = texture(screen_texture, frag_uv).rgb;
    FragColor = vec4(color,1);
}