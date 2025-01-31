#pragma once

#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>

namespace atcg
{
class OptixComponent
{
public:
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    ATCG_INLINE void ensureInitialized(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
    {
        if(!_initialized) initializePipeline(pipeline, sbt);
    }

private:
    bool _initialized = false;
};
}    // namespace atcg