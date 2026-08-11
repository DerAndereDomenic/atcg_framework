#pragma once

#include <Core/API.h>
#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>

namespace atcg
{
/**
 * @brief An Optix component is a part of a raytracing pipeline
 */
struct ATCG_API RaytracingComponent
{
public:
    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

    /**
     * @brief Initialize the component in the raytracing pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    /**
     * @brief Ensure the component is initialized in the raytracing pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    ATCG_INLINE virtual void ensureInitialized(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
    {
        if(!_initialized)
        {
            initializePipeline(pipeline, sbt);
            _initialized = true;
        }
    }

    /**
     * @brief Check if the component is initialized in the raytracing pipeline
     *
     * @return True if the component is initialized, false otherwise
     */
    ATCG_INLINE bool isInitialized() const { return _initialized; }

    /**
     * @brief Mark the component as initialized in the raytracing pipeline
     */
    ATCG_INLINE void markInitialized() { _initialized = true; }

private:
    bool _initialized = false;
};
}    // namespace atcg