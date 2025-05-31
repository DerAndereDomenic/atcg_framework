#pragma once

#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>

namespace atcg
{
/**
 * @brief An Optix component is a part of a raytracing pipeline
 */
class OptixComponent
{
public:
    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    /**
     * @brief Ensure that the pipeline is initialized
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    ATCG_INLINE void ensureInitialized(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
    {
        if(!_initialized) initializePipeline(pipeline, sbt);
    }

private:
    bool _initialized = false;
};
}    // namespace atcg