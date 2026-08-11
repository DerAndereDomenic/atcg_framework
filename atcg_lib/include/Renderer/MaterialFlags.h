#pragma once

namespace atcg
{
enum class MaterialFlag : uint32_t
{
    None                = 0x00,
    IdealReflection     = 0x01,
    GlossyReflection    = 0x02,
    DiffuseReflection   = 0x04,
    IdealTransmission   = 0x08,
    GlossyTransmission  = 0x10,
    DiffuseTransmission = 0x20,

    NullTransmission = 0x40,

    AnyDelta        = IdealReflection | IdealTransmission | NullTransmission,
    AnyReflection   = IdealReflection | GlossyReflection | DiffuseReflection,
    AnyTransmission = IdealTransmission | GlossyTransmission | DiffuseTransmission | NullTransmission,
    Any             = AnyReflection | AnyTransmission,
};

ATCG_HOST_DEVICE inline MaterialFlag operator~(MaterialFlag a)
{
    return (MaterialFlag) ~(int)a;
}
ATCG_HOST_DEVICE inline MaterialFlag operator|(MaterialFlag a, MaterialFlag b)
{
    return (MaterialFlag)((int)a | (int)b);
}
ATCG_HOST_DEVICE inline MaterialFlag operator&(MaterialFlag a, MaterialFlag b)
{
    return (MaterialFlag)((int)a & (int)b);
}
ATCG_HOST_DEVICE inline MaterialFlag operator^(MaterialFlag a, MaterialFlag b)
{
    return (MaterialFlag)((int)a ^ (int)b);
}
ATCG_HOST_DEVICE inline MaterialFlag& operator|=(MaterialFlag& a, MaterialFlag b)
{
    return (MaterialFlag&)((int&)a |= (int)b);
}
ATCG_HOST_DEVICE inline MaterialFlag& operator&=(MaterialFlag& a, MaterialFlag b)
{
    return (MaterialFlag&)((int&)a &= (int)b);
}
ATCG_HOST_DEVICE inline MaterialFlag& operator^=(MaterialFlag& a, MaterialFlag b)
{
    return (MaterialFlag&)((int&)a ^= (int)b);
}

ATCG_HOST_DEVICE inline bool hasMaterialFlag(MaterialFlag flags, MaterialFlag flag)
{
    return (flags & flag) != MaterialFlag::None;
}
}    // namespace atcg