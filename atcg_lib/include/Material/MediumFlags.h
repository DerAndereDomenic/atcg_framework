#pragma once

namespace atcg
{
enum class MediumFlag : uint32_t
{
    None          = 0x00,
    Homogeneous   = 0x01,
    Heterogeneous = 0x02,

    Any = Homogeneous | Heterogeneous,
};

ATCG_HOST_DEVICE inline MediumFlag operator~(MediumFlag a)
{
    return (MediumFlag) ~(int)a;
}
ATCG_HOST_DEVICE inline MediumFlag operator|(MediumFlag a, MediumFlag b)
{
    return (MediumFlag)((int)a | (int)b);
}
ATCG_HOST_DEVICE inline MediumFlag operator&(MediumFlag a, MediumFlag b)
{
    return (MediumFlag)((int)a & (int)b);
}
ATCG_HOST_DEVICE inline MediumFlag operator^(MediumFlag a, MediumFlag b)
{
    return (MediumFlag)((int)a ^ (int)b);
}
ATCG_HOST_DEVICE inline MediumFlag& operator|=(MediumFlag& a, MediumFlag b)
{
    return (MediumFlag&)((int&)a |= (int)b);
}
ATCG_HOST_DEVICE inline MediumFlag& operator&=(MediumFlag& a, MediumFlag b)
{
    return (MediumFlag&)((int&)a &= (int)b);
}
ATCG_HOST_DEVICE inline MediumFlag& operator^=(MediumFlag& a, MediumFlag b)
{
    return (MediumFlag&)((int&)a ^= (int)b);
}

ATCG_HOST_DEVICE inline bool hasMediumFlag(MediumFlag flags, MediumFlag flag)
{
    return (flags & flag) != MediumFlag::None;
}
}    // namespace atcg