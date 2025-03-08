#version 430 core
out vec4 FragColor;
out int entityID;

in vec3 frag_tex;

uniform samplerCube skybox;

void main()
{
    vec3 color = texture(skybox, frag_tex).rgb;
    FragColor = vec4(pow(vec3(1) - exp(-color), vec3(1.0 / 2.4)), 1.0);
    entityID = -1;
}