#include <Renderer/RenderPasses/ForwardPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
ForwardPass::ForwardPass() : RenderPass("ForwardPass")
{
    registerOutput("framebuffer", nullptr);
    setSetupFunction(
        [this](Dictionary& context, Dictionary& data, Dictionary& output)
        {
            auto renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());
            data.setValue("dummy_skybox", atcg::make_ref<Skybox>());
        });


    setRenderFunction(
        [](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary&)
        {
            auto renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());
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

            auto skybox     = inputs.getValueOr<atcg::ref_ptr<Skybox>>("skybox", nullptr);
            bool has_skybox = context.getValueOr("has_skybox", false) && (skybox != nullptr);

            Dictionary auxiliary;
            auxiliary.setValue("point_light_depth_maps", point_light_depth_maps);
            auxiliary.setValue("skybox", has_skybox ? skybox : data.getValue<atcg::ref_ptr<Skybox>>("dummy_skybox"));
            auxiliary.setValue("has_skybox", has_skybox);

            for(auto e: view)
            {
                Entity entity(e, scene);
                if(entity.hasComponent<CustomRenderComponent>())
                {
                    CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
                    renderer.callback(entity, camera);
                }

                ComponentRegistry::renderAllComponents(renderer, entity, camera, auxiliary);
            }
        });
}
}    // namespace atcg