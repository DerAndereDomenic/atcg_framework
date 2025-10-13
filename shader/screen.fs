#version 400 core

layout(location = 0) out vec4 FragColor;
layout(location = 1) out int entityID;

in vec2 frag_uv;

uniform sampler2D screen_texture;
uniform isampler2D entity_ids;

subroutine int _getEntityID();

subroutine(_getEntityID) int getDefaultID()
{
    return -1;
}

subroutine(_getEntityID) int getFromTextureID()
{
    return texture(entity_ids, frag_uv).r;
}

subroutine uniform _getEntityID getEntityID;

void main()
{
    vec3 color = texture(screen_texture, frag_uv).rgb;
    int entity_id = getEntityID();
    FragColor = vec4(color, 1);
    entityID = entity_id;
}