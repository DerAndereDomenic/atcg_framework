#version 450 core

layout(location = 0) out vec4 out_color;
layout(location = 1) out int out_entityID;
layout(location = 2) out uint out_stencil;

in vec2 frag_uv;

uniform sampler2DMS in_color;
uniform isampler2DMS in_entity;
uniform usampler2DMS in_stencil;
uniform sampler2DMS in_depth;

void main()
{
    ivec2 size = textureSize(in_color);
    ivec2 texel = ivec2(frag_uv * vec2(size));

    int samples = textureSamples(in_color);

    vec4 color = vec4(0.0);
    int entityID = 0;
    uint stencil = 0;
    float depth = 1.0;

    for(int i = 0; i < samples; i++)
    {
        color += texelFetch(in_color, texel, i);
        depth = min(depth, texelFetch(in_depth, texel, i).r);
    }
    color /= float(samples);

    entityID = texelFetch(in_entity, texel, 0).r;
    stencil = texelFetch(in_stencil, texel, 0).r;

    out_color = color;
    out_entityID = entityID;
    out_stencil = stencil;

    gl_FragDepth = depth;
}