#pragma once

namespace atcg
{
enum class EmitterFlags : uint32_t
{
    None = 0x00,
    // This emitter is infinitely far away from the scene
    DistantEmitter = 0x01,
    // This emitter has no surface area (i.e. point light) and can therefore not be encountered by regular ray tracing
    InfinitesimalSize = 0x02
};

ATCG_HOST_DEVICE inline EmitterFlags operator~(EmitterFlags a)
{
    return (EmitterFlags) ~(int)a;
}
ATCG_HOST_DEVICE inline EmitterFlags operator|(EmitterFlags a, EmitterFlags b)
{
    return (EmitterFlags)((int)a | (int)b);
}
ATCG_HOST_DEVICE inline EmitterFlags operator&(EmitterFlags a, EmitterFlags b)
{
    return (EmitterFlags)((int)a & (int)b);
}
ATCG_HOST_DEVICE inline EmitterFlags operator^(EmitterFlags a, EmitterFlags b)
{
    return (EmitterFlags)((int)a ^ (int)b);
}
ATCG_HOST_DEVICE inline EmitterFlags& operator|=(EmitterFlags& a, EmitterFlags b)
{
    return (EmitterFlags&)((int&)a |= (int)b);
}
ATCG_HOST_DEVICE inline EmitterFlags& operator&=(EmitterFlags& a, EmitterFlags b)
{
    return (EmitterFlags&)((int&)a &= (int)b);
}
ATCG_HOST_DEVICE inline EmitterFlags& operator^=(EmitterFlags& a, EmitterFlags b)
{
    return (EmitterFlags&)((int&)a ^= (int)b);
}
}    // namespace atcg