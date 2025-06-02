#version 330 core

#define PI 3.14159

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outBaseColor;
layout (location = 3) out vec4 outSpecMet;
layout (location = 4) out int outEntityID;

in vec3 frag_normal;
in vec3 frag_pos;
in vec3 frag_color;
in vec2 frag_uv;
in mat3 frag_tbn;

uniform vec3 flat_color;
uniform int entityID;

// Material textures
uniform sampler2D texture_diffuse;
uniform sampler2D texture_normal;
uniform sampler2D texture_roughness;
uniform sampler2D texture_metallic;

void main()
{
    // Get parameters
    vec4 diffuse_lookup = texture(texture_diffuse, frag_uv);
    vec3 color_diffuse = frag_color * flat_color * diffuse_lookup.rgb;
    float roughness = texture(texture_roughness, frag_uv).r;
    float metallic = texture(texture_metallic, frag_uv).r;
    vec3 texture_normal = normalize(texture(texture_normal, frag_uv).rgb * 2.0 - 1.0);

    // PBR material shader
    vec3 normal = frag_tbn * texture_normal;
    
    outPosition = vec4(frag_pos, 1);
    outNormal = vec4(normal, 1);
    outBaseColor = vec4(color_diffuse, 1);
    outSpecMet = vec4(metallic, roughness, 0, 1);
    outEntityID = entityID;
}