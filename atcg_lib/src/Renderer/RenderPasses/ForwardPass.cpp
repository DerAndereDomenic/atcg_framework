#include <Renderer/RenderPasses/ForwardPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
ForwardPass::ForwardPass(const RenderTargetDesc& desc) : RenderPass(desc, "ForwardPass")
{
    registerOutput("framebuffer", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
    setSetupFunction(
        [this](Dictionary& context, Dictionary& data, Dictionary& output)
        {
            data.setValue("dummy_skybox", atcg::make_ref<Skybox>());
            if(_render_target.mode == RenderTargetMode::RENDER_TARGET_OWN_FRAMEBUFFER)
            {
                data.setValue("target", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
            }
        });


    setRenderFunction(
        [this](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary& outputs)
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

            auto skybox     = *inputs.getValueOr<atcg::ref_ptr<atcg::ref_ptr<Skybox>>>("skybox", nullptr);
            bool has_skybox = context.getValueOr("has_skybox", false) && (skybox != nullptr);

            Dictionary auxiliary;
            auxiliary.setValue("point_light_depth_maps", point_light_depth_maps);
            auxiliary.setValue("skybox", has_skybox ? skybox : data.getValue<atcg::ref_ptr<Skybox>>("dummy_skybox"));
            auxiliary.setValue("has_skybox", has_skybox);

            auto output_framebuffer = outputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffe"
                                                                                                  "r");
            auto target             = prepareFramebuffer(context, inputs, data, outputs);
            *output_framebuffer     = target;

            if(_render_target.clear)
            {
                renderer->clear();
                // We assume that this is an entity buffer, better solution?
                if(target->numColorAttachements() > 1 &&
                   target->getColorAttachement(1)->getSpecification().format == TextureFormat::RINT)
                {
                    int value = -1;
                    target->getColorAttachement(1)->fill(&value);
                }
            }
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