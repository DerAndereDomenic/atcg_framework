#version 330 core

layout(location = 0) out vec4 FragColor;
layout(location = 1) out int outEntityID;

in vec2 frag_uv;

uniform sampler2D screen_texture;
uniform int entityID;

void main()
{
    vec3 color = texture(screen_texture, frag_uv).rgb;
    FragColor = vec4(color, 1);
    outEntityID = entityID;
}