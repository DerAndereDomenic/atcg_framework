#pragma once

namespace atcg
{
enum class BSDFComponentType : uint32_t
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

ATCG_HOST_DEVICE inline BSDFComponentType operator~(BSDFComponentType a)
{
    return (BSDFComponentType) ~(int)a;
}
ATCG_HOST_DEVICE inline BSDFComponentType operator|(BSDFComponentType a, BSDFComponentType b)
{
    return (BSDFComponentType)((int)a | (int)b);
}
ATCG_HOST_DEVICE inline BSDFComponentType operator&(BSDFComponentType a, BSDFComponentType b)
{
    return (BSDFComponentType)((int)a & (int)b);
}
ATCG_HOST_DEVICE inline BSDFComponentType operator^(BSDFComponentType a, BSDFComponentType b)
{
    return (BSDFComponentType)((int)a ^ (int)b);
}
ATCG_HOST_DEVICE inline BSDFComponentType& operator|=(BSDFComponentType& a, BSDFComponentType b)
{
    return (BSDFComponentType&)((int&)a |= (int)b);
}
ATCG_HOST_DEVICE inline BSDFComponentType& operator&=(BSDFComponentType& a, BSDFComponentType b)
{
    return (BSDFComponentType&)((int&)a &= (int)b);
}
ATCG_HOST_DEVICE inline BSDFComponentType& operator^=(BSDFComponentType& a, BSDFComponentType b)
{
    return (BSDFComponentType&)((int&)a ^= (int)b);
}
}    // namespace atcg