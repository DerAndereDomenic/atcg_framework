#pragma once

#include <Scene/OptixScene.h>
#include <Scene/Scene.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <Core/RaytracingContext.h>

namespace atcg
{
/**
 * @brief A class to convert all scene elements into the optix representation
 */
class SceneAdapter
{
public:
    /**
     * @brief Constructor
     *
     * @param context The context
     * @param pipeline The pipeline
     * @param sbt The SBT
     */
    SceneAdapter(const atcg::ref_ptr<RaytracingContext>& context,
                 const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                 const atcg::ref_ptr<ShaderBindingTable>& sbt)
        : _context(context),
          _pipeline(pipeline),
          _sbt(sbt)
    {
    }

    /**
     * @brief Apply the conversion
     *
     * @param scene The scene to convert
     *
     * @return The converted scene
     */
    atcg::ref_ptr<OptixScene> apply(const atcg::ref_ptr<Scene>& scene);

private:
    template<typename T>
    void prepareComponent(const atcg::ref_ptr<OptixScene>& result, Entity entity);

private:
    atcg::ref_ptr<RaytracingContext> _context;
    atcg::ref_ptr<RayTracingPipeline> _pipeline;
    atcg::ref_ptr<ShaderBindingTable> _sbt;
};
}    // namespace atcg