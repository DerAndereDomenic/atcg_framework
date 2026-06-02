#pragma once

#include <optix.h>

#include <Core/API.h>
#include <Core/Platform.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <Core/OptixComponent.h>
#include <Core/RaytracingContext.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>
#include <DataStructure/TorchUtils.h>
#include <Plugin/Plugin.h>

#include <vector>

namespace atcg
{
/**
 * @brief A class to model an integrator
 */
class ATCG_API Integrator
{
public:
    using PluginCreate =
        std::function<std::shared_ptr<Integrator>(const atcg::ref_ptr<RaytracingContext>&, const atcg::Dictionary&)>;
    ATCG_PLUGIN_BASE_CLASS(Integrator);

    /**
     * @brief Constructor
     *
     * @param context The raytracing context
     * @param dict Additional paramaters
     */
    Integrator(const atcg::ref_ptr<RaytracingContext>& context, const atcg::Dictionary& dict) : _context(context)
    {
        _pipeline = atcg::make_ref<RayTracingPipeline>(context, dict.getValueOr<uint32_t>("num_rays", 1));
        _sbt      = atcg::make_ref<ShaderBindingTable>();
    }

    /**
     * @brief Destructor
     */
    virtual ~Integrator() = default;

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

    /**
     * @brief Generate the rays and write output
     *
     * @param in_out_dictionary The input and output data
     */
    virtual void generateRays(Dictionary& in_out_dictionary) = 0;

    /**
     * @brief Reset the internal structure of the integrator
     */
    virtual void reset() = 0;

    /**
     * @brief Get the pipeline
     *
     * @return The pipeline
     */
    ATCG_INLINE atcg::ref_ptr<RayTracingPipeline> getPipeline() const { return _pipeline; }

    /**
     * @brief Get the shader binding table
     *
     * @return The SBT
     */
    ATCG_INLINE atcg::ref_ptr<ShaderBindingTable> getSBT() const { return _sbt; }

protected:
    atcg::ref_ptr<RaytracingContext> _context;

    atcg::ref_ptr<RayTracingPipeline> _pipeline;
    atcg::ref_ptr<ShaderBindingTable> _sbt;
};
}    // namespace atcg