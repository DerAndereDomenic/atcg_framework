#pragma once

namespace atcg
{
struct TraceParameters
{
    uint32_t rayFlags;
    uint32_t SBToffset;
    uint32_t SBTstride;
    uint32_t missSBTIndex;
};
}    // namespace atcg