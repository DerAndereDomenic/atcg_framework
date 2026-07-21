#pragma once

#include <Core/API.h>
#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
/**
 * @brief An Optix component is a part of a raytracing pipeline
 */
class ATCG_API OptixComponent
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

class ATCG_API Differentiable
{
public:
    virtual void markParametersAsOptimizable(const std::string& parameter_name) = 0;

    virtual void clampParameters() = 0;

    ATCG_INLINE void markParametersAsOptimizable(const std::vector<std::string>& parameter_names)
    {
        for(const auto& name: parameter_names)
        {
            markParametersAsOptimizable(name);
        }
    }

    ATCG_INLINE torch::Tensor& getParameter(const std::string& name)
    {
        auto it = _parameter_map.find(name);
        if(it == _parameter_map.end())
        {
            ATCG_ERROR("Parameter {} not found in differentiable component", name);
        }
        return _parameters[it->second];
    }

    ATCG_INLINE torch::Tensor& getGradient(const std::string& name)
    {
        auto it = _parameter_map.find(name);
        if(it == _parameter_map.end())
        {
            ATCG_ERROR("Parameter {} not found in differentiable component", name);
        }
        return _gradients[it->second];
    }

    ATCG_INLINE void setParameter(const std::string& name, const torch::Tensor& value)
    {
        auto it = _parameter_map.find(name);
        if(it == _parameter_map.end())
        {
            size_t index         = _parameters.size();
            _parameter_map[name] = index;
            _parameters.push_back(value);
            _gradients.push_back(torch::zeros_like(value));
        }
        else
        {
            _parameters[it->second] = value;
            _gradients[it->second]  = torch::zeros_like(value);
        }
    }

    ATCG_INLINE void zeroGradientBuffers()
    {
        for(auto& grad: _gradients)
        {
            grad.zero_();
        }
    }

    ATCG_INLINE std::vector<torch::Tensor> getOptimizableParameterList() const
    {
        std::vector<torch::Tensor> optimizable_parameters;
        for(const auto& [name, index]: _parameter_map)
        {
            if(isParameterOptimizable(name))
            {
                optimizable_parameters.push_back(_parameters[index]);
            }
        }
        return optimizable_parameters;
    }

    ATCG_INLINE std::vector<torch::Tensor> getOptimizableGradientList() const
    {
        std::vector<torch::Tensor> optimizable_gradients;
        for(const auto& [name, index]: _parameter_map)
        {
            if(isParameterOptimizable(name))
            {
                optimizable_gradients.push_back(_gradients[index]);
            }
        }
        return optimizable_gradients;
    }

    ATCG_INLINE bool hasParameter(const std::string& name) const
    {
        return _parameter_map.find(name) != _parameter_map.end();
    }

    ATCG_INLINE bool isParameterOptimizable(const std::string& name) const
    {
        auto it = _parameter_map.find(name);
        if(it == _parameter_map.end())
        {
            return false;
        }
        return _parameters[it->second].requires_grad();
    }

private:
    std::vector<torch::Tensor> _parameters;
    std::vector<torch::Tensor> _gradients;
    std::unordered_map<std::string, size_t> _parameter_map;
};
}    // namespace atcg