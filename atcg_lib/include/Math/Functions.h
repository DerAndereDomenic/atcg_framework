#pragma once

namespace atcg
{
namespace Math
{
template<typename T>
T ceil_div(T num, T den)
{
    return (num - T(1)) / den + T(1);
}

}    // namespace Math

}    // namespace atcg

#include "../../src/Math/FunctionsDetail.h"