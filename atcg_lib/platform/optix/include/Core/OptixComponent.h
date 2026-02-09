#pragma once

#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
/**
 * @brief An Optix component is a part of a raytracing pipeline
 */
class OptixComponent
{
public:
    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    ATCG_INLINE virtual void ensureInitialized(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
    {
        if(!_initialized)
        {
            initializePipeline(pipeline, sbt);
            _initialized = true;
        }
    }

    ATCG_INLINE bool isInitialized() const { return _initialized; }

    ATCG_INLINE void markInitialized() { _initialized = true; }

private:
    bool _initialized = false;
};

class Differentiable
{
public:
    virtual std::vector<torch::Tensor> getParameters() const = 0;

    virtual std::vector<torch::Tensor> getParameterGradients() const = 0;

    virtual void zeroGrad() = 0;

    virtual void markOptimizable() = 0;

    bool isOptimizable() const { return _optimizable; }

    virtual void clampParameters() = 0;

protected:
    bool _optimizable = false;
};
}    // namespace atcg