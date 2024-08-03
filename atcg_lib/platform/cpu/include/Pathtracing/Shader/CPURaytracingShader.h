#pragma once

#include <Pathtracing/RaytracingShader.h>

namespace atcg
{
class CPURaytracingShader : public RaytracingShader
{
public:
    CPURaytracingShader() = default;

    virtual ~CPURaytracingShader() {}

    virtual void initializePipeline() = 0;

    virtual void reset() = 0;

    virtual void generateRays(torch::Tensor& output) = 0;
};
}    // namespace atcg