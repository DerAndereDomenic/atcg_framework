#pragma once

namespace atcg
{
enum class PhaseFlag : uint32_t
{
    None        = 0x00,
    Isotropic   = 0x01,
    Anisotropic = 0x02,
    Delta       = 0x04,

    AnyDelta = Delta,

    Any = Isotropic | Anisotropic | Delta,
};

ATCG_HOST_DEVICE inline PhaseFlag operator~(PhaseFlag a)
{
    return (PhaseFlag) ~(int)a;
}
ATCG_HOST_DEVICE inline PhaseFlag operator|(PhaseFlag a, PhaseFlag b)
{
    return (PhaseFlag)((int)a | (int)b);
}
ATCG_HOST_DEVICE inline PhaseFlag operator&(PhaseFlag a, PhaseFlag b)
{
    return (PhaseFlag)((int)a & (int)b);
}
ATCG_HOST_DEVICE inline PhaseFlag operator^(PhaseFlag a, PhaseFlag b)
{
    return (PhaseFlag)((int)a ^ (int)b);
}
ATCG_HOST_DEVICE inline PhaseFlag& operator|=(PhaseFlag& a, PhaseFlag b)
{
    return (PhaseFlag&)((int&)a |= (int)b);
}
ATCG_HOST_DEVICE inline PhaseFlag& operator&=(PhaseFlag& a, PhaseFlag b)
{
    return (PhaseFlag&)((int&)a &= (int)b);
}
ATCG_HOST_DEVICE inline PhaseFlag& operator^=(PhaseFlag& a, PhaseFlag b)
{
    return (PhaseFlag&)((int&)a ^= (int)b);
}

ATCG_HOST_DEVICE inline bool hasPhaseFlag(PhaseFlag flags, PhaseFlag flag)
{
    return (flags & flag) != PhaseFlag::None;
}
}    // namespace atcg