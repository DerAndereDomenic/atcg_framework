#pragma once

namespace atcg
{
/**
 * @brief Calculate the surface area of a triangle with edge lengths a b c
 *
 * @param a Edge length
 * @param b Edge length
 * @param c Edge length
 * @return T The area
 */
template<typename T>
T areaFromMetric(T a, T b, T c)
{
    // Numerically stable herons formula for area of triangle with side lengths a, b and c.
    if(a < b) std::swap(a, b);
    if(a < c) std::swap(a, c);
    if(b < c) std::swap(b, c);

    T p = std::sqrt(std::abs((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)))) / 4;
    return p;
}

namespace Math
{
/**
 *  @brief Evaluate a spherical harmonic basis function
 *
 *  @tparam The data type
 *  @param l Degree of the harmonic l >= 0
 *  @param m Order of the harmonic |m| <= l
 *  @param phi Angle between [0, 2pi]
 *  @param theta Angle between [0, pi]
 *  @return The evaluated function
 */
template<typename T>
T sphericalHarmonic(const int32_t& l, const int32_t& m, const T& phi, const T& theta);
}    // namespace Math

}    // namespace atcg

#include "../../src/Math/FunctionsDetail.h"