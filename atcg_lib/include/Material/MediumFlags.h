#pragma once

namespace atcg
{
enum class MediumFlags : uint32_t
{
    None          = 0x00,
    Homogeneous   = 0x01,
    Heterogeneous = 0x02,

    Any = Homogeneous | Heterogeneous,
};

ATCG_HOST_DEVICE inline MediumFlags operator~(MediumFlags a)
{
    return (MediumFlags) ~(int)a;
}
ATCG_HOST_DEVICE inline MediumFlags operator|(MediumFlags a, MediumFlags b)
{
    return (MediumFlags)((int)a | (int)b);
}
ATCG_HOST_DEVICE inline MediumFlags operator&(MediumFlags a, MediumFlags b)
{
    return (MediumFlags)((int)a & (int)b);
}
ATCG_HOST_DEVICE inline MediumFlags operator^(MediumFlags a, MediumFlags b)
{
    return (MediumFlags)((int)a ^ (int)b);
}
ATCG_HOST_DEVICE inline MediumFlags& operator|=(MediumFlags& a, MediumFlags b)
{
    return (MediumFlags&)((int&)a |= (int)b);
}
ATCG_HOST_DEVICE inline MediumFlags& operator&=(MediumFlags& a, MediumFlags b)
{
    return (MediumFlags&)((int&)a &= (int)b);
}
ATCG_HOST_DEVICE inline MediumFlags& operator^=(MediumFlags& a, MediumFlags b)
{
    return (MediumFlags&)((int&)a ^= (int)b);
}

ATCG_HOST_DEVICE inline bool hasMediumFlag(MediumFlags flags, MediumFlags flag)
{
    return (flags & flag) != MediumFlags::None;
}
}    // namespace atcg