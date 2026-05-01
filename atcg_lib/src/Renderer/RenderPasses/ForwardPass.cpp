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
            auto scene       = context.getValue<atcg::ref_ptr<Scene>>("scene");
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

            auto skybox = *inputs.getValueOr<atcg::ref_ptr<atcg::ref_ptr<Skybox>>>(
                "skybox",
                atcg::make_ref<atcg::ref_ptr<Skybox>>(nullptr));
            bool has_skybox = context.getValueOr("has_skybox", false) && (skybox != nullptr);

            Dictionary auxiliary;
            auxiliary.setValue("point_light_depth_maps", point_light_depth_maps);
            auxiliary.setValue("skybox", has_skybox ? skybox : data.getValue<atcg::ref_ptr<Skybox>>("dummy_skybox"));
            auxiliary.setValue("has_skybox", has_skybox);
            auxiliary.setValue("draw_cameras", context.getValueOr("draw_cameras", true));
            if(inputs.contains("depth_buffer"))
            {
                auto depth_map = *inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("depth_buffer");
                auxiliary.setValue("depth_map", depth_map->getDepthAttachement());
            }

            auto output_framebuffer = outputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffe"
                                                                                                  "r");
            auto target             = prepareFramebuffer(context, inputs, data, outputs);
            *output_framebuffer     = target;

            GraphicsCommand::beginRenderPass(target);
            if(_render_target.clear)
            {
                GraphicsCommand::clear();
                //  We assume that this is an entity buffer, better solution?
                if(target->numColorAttachements() > 1 &&
                   target->getColorAttachement(1)->getSpecification().format == TextureFormat::RINT)
                {
                    int value = -1;
                    target->getColorAttachement(1)->fill(&value);
                }

                if(target->numColorAttachements() > 2 &&
                   target->getColorAttachement(2)->getSpecification().format == TextureFormat::RINT8)
                {
                    uint8_t value = 0;
                    target->getColorAttachement(2)->fill(&value);
                }
            }

            this->_transparent_entities.clear();
            for(auto e: view)
            {
                Entity entity(e, scene.get());
                if(entity.hasComponent<CustomRenderComponent>())
                {
                    CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
                    renderer.callback(entity, camera);
                }

                if(entity.hasAnyComponent<TransparencyComponent>() &&
                   entity.getComponent<TransparencyComponent>().transparent)
                {
                    TransformComponent& transform = entity.getComponent<TransformComponent>();
                    glm::vec3 position            = transform.getPosition();
                    if(entity.hasComponent<GeometryComponent>())
                    {
                        GeometryComponent& geometry = entity.getComponent<GeometryComponent>();

                        BoundingBox bbox = geometry.graph()->getBoundingBox();
                        bbox             = Utils::transformBoundingBox(bbox, transform.getModel());

                        position = (bbox.min + bbox.max) * 0.5f;
                    }

                    float distance_to_camera = glm::length(camera->getPosition() - position);

                    this->_transparent_entities.push_back({entity, distance_to_camera});
                    continue;
                }

                ComponentRegistry::renderAllComponents(renderer, entity, camera, auxiliary);
            }

            std::sort(this->_transparent_entities.begin(),
                      this->_transparent_entities.end(),
                      [](const TransparentRenderData& a, const TransparentRenderData& b)
                      { return a.distance_to_camera > b.distance_to_camera; });

            for(const auto& data: this->_transparent_entities)
            {
                ComponentRegistry::renderAllComponents(renderer, data.entity, camera, auxiliary);
            }

            GraphicsCommand::endRenderPass();
        });
}
}    // namespace atcg