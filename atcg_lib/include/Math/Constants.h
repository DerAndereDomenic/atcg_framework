#pragma once

#include <Core/glm.h>

namespace atcg
{
namespace Constants
{
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T zero()
{
    return T(0);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T one()
{
    return T(1);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T pi()
{
    return glm::pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T two_pi()
{
    return glm::two_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T root_pi()
{
    return glm::root_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T half_pi()
{
    return glm::half_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T three_over_two_pi()
{
    return glm::three_over_two_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T quarter_pi()
{
    return glm::quarter_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T one_over_pi()
{
    return glm::one_over_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T one_over_two_pi()
{
    return glm::one_over_two_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T two_over_pi()
{
    return glm::two_over_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T four_over_pi()
{
    return glm::four_over_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T two_over_root_pi()
{
    return glm::two_over_root_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T one_over_root_two()
{
    return glm::one_over_root_two<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T root_half_pi()
{
    return glm::root_half_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T root_two_pi()
{
    return glm::root_two_pi<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T root_ln_four()
{
    return glm::root_ln_four<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T e()
{
    return glm::e<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T euler()
{
    return glm::euler<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T root_two()
{
    return glm::root_two<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T root_three()
{
    return glm::root_three<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T root_five()
{
    return glm::root_five<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T ln_two()
{
    return glm::ln_two<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T ln_ten()
{
    return glm::ln_ten<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T ln_ln_two()
{
    return glm::ln_ln_two<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T half()
{
    return T(0.5);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T third()
{
    return glm::third<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T two_thirds()
{
    return glm::two_thirds<T>();
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T golden_ratio()
{
    return glm::golden_ratio<T>();
}

//	k_B, the Boltzmann constant (J⋅K⁻¹)
//		Value is given to known precision.
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T boltzmann()
{
    return T(1.38064852e-23L);
}

//	h, the Planck constant (J⋅s)
//		Value is a precise definition.
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T h()
{
    return T(6.62607015e-34L);
}

//	c, the speed of light (m⋅s⁻¹)
//		Value is a precise definition.
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T c()
{
    return T(299'792'458.0L);
}

//  G, Newtonian constant of gravitation (m3⋅kg−1⋅s−2)
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T G()
{
    return T(6.6743015e-11);
}

//  g, Gravitation acceleration on earth (m⋅s-2)
template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE ATCG_CONSTEXPR T g()
{
    return T(9.80665);
}
}    // namespace Constants
}    // namespace atcg