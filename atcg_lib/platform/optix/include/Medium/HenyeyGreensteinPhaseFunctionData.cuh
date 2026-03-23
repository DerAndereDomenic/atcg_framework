#pragma once

namespace atcg
{
struct HenyeyGreensteinPhaseFunctionData
{
    float* g;

    float* g_grad;

    bool optimize_g = false;
};
}    // namespace atcg