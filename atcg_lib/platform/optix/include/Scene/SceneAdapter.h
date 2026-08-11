#pragma once

#include <Core/glm.h>
#include <Core/API.h>
#include <Scene/OptixScene.h>
#include <Scene/Scene.h>
#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>
#include <Renderer/RaytracingContext.h>

namespace atcg
{
/**
 * @brief A class to convert all scene elements into the optix representation
 */
class ATCG_API SceneAdapter
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
        _scene_aabb.min = glm::vec3(std::numeric_limits<float>::max());
        _scene_aabb.max = glm::vec3(std::numeric_limits<float>::lowest());
    }

    /**
     * @brief Apply the conversion
     *
     * @param scene The scene to convert
     *
     * @return The converted scene
     */
    atcg::ref_ptr<OptixScene> apply(const atcg::ref_ptr<Scene>& scene, const uint32_t width, const uint32_t height);

    atcg::BoundingBox getSceneAABB() const { return _scene_aabb; }

private:
    template<typename T>
    void prepareComponent(const atcg::ref_ptr<OptixScene>& result, Entity entity);

private:
    atcg::ref_ptr<RaytracingContext> _context;
    atcg::ref_ptr<RayTracingPipeline> _pipeline;
    atcg::ref_ptr<ShaderBindingTable> _sbt;

    std::unordered_map<AssetHandle, atcg::ref_ptr<Shape>> _shape_cache;
    std::unordered_map<AssetHandle, atcg::ref_ptr<BSDF>> _bsdf_cache;

    atcg::BoundingBox _scene_aabb;
};
}    // namespace atcg