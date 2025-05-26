#include <Renderer/RenderPasses/ForwardPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{
ForwardPass::ForwardPass(const atcg::ref_ptr<Skybox>& skybox) : RenderPass("ForwardPass"), _skybox(skybox)
{
    if(!_skybox)
    {
        _skybox = atcg::make_ref<Skybox>();
    }
    _data.setValue("skybox", _skybox);

    setRenderFunction(
        [](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary&)
        {
            auto scene       = context.getValue<Scene*>("scene");
            auto camera      = context.getValue<atcg::ref_ptr<Camera>>("camera");
            const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

            atcg::ref_ptr<atcg::TextureCubeArray> point_light_depth_maps = nullptr;

            if(inputs.contains("point_light_depth_maps"))
            {
                point_light_depth_maps = *inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<atcg::TextureCubeArray>>>("point_"
                                                                                                                "light_"
                                                                                                                "depth_"
                                                                                                                "maps");
            }

            Dictionary auxiliary;
            auxiliary.setValue("point_light_depth_maps", point_light_depth_maps);
            auxiliary.setValue("skybox", data.getValue<atcg::ref_ptr<Skybox>>("skybox"));
            auxiliary.setValue("has_skybox", context.getValueOr<bool>("has_skybox", false));

            ComponentRenderer component_renderer;
            for(auto e: view)
            {
                Entity entity(e, scene);
                if(entity.hasComponent<CustomRenderComponent>())
                {
                    CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
                    renderer.callback(entity, camera);
                }

                component_renderer.renderComponent<MeshRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<PointRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<PointSphereRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<EdgeRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<EdgeCylinderRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<InstanceRenderComponent>(entity, camera, auxiliary);
            }
        });
}
}    // namespace atcg