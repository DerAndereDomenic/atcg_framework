#pragma once

#include <Pathtracing/BSDFFlags.h>

namespace atcg
{
class BSDF
{
public:
    BSDF() = default;

    virtual ~BSDF() {}

protected:
    BSDFComponentType _flags;
};
}    // namespace atcg