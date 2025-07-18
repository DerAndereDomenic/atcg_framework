#version 460 core

#define inf 1e12

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(r32f, binding = 0) uniform image2D texture;

uniform int num_points;

uint tea(uint val0, uint val1)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;
    for (uint n = 0; n < 3; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

uint prev;
float rnd()
{
    uint LCG_A = 1664525u;
    uint LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return (float(prev & 0x00FFFFFF) / float(0x01000000));
}

void main()
{
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);

    ivec2 size = imageSize(texture);

    uint tid = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * size.x;
    // All threads should use the same seed
    prev = tea(1337, 420);

    vec2 uv = vec2(texelCoord) / size;
	
    float value = inf;
    for(uint i = 0; i < num_points; ++i)
    {
        // Random point in plane
        float x = rnd();
        float y = rnd();

        // Periodic
        float dx = x - uv.x;
        float dy = y - uv.y;

        if(dx > 0.5) dx -= 1.0;
        if(dx <= -0.5) dx += 1.0;
        if(dy > 0.5) dy -= 1.0;
        if(dy <= -0.5) dy += 1.0;

        float dist = sqrt(dx * dx + dy * dy);
        if(dist < value) value = dist;
    }

    imageStore(texture, texelCoord, vec4(value));
}