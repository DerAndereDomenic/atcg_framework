#include <Renderer/RenderPasses/ForwardPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{

ForwardPass::ForwardPass(Dictionary& properties) : RenderPass(properties, "ForwardPass") {}

RenderPassReflection ForwardPass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;

    TextureType type = ctx.num_samples > 1 ? TextureType::TEXTURE_2D_MULTISAMPLE : TextureType::TEXTURE_2D;

    reflection.addInput("depth_buffer");
    reflection.addInput("point_light_depth_maps");

    ResourceDescription desc_color;
    desc_color.type           = ResourceType::Texture;
    desc_color.texture.type   = type;
    desc_color.texture.format = TextureFormat::RGBAFLOAT;

    uint32_t color_handle = reflection.addOutput("output", desc_color);

    ResourceDescription desc_depth;
    desc_depth.type           = ResourceType::Texture;
    desc_depth.texture.type   = type;
    desc_depth.texture.format = TextureFormat::DEPTH;

    uint32_t depth_handle = reflection.addOutput("out_depth_buffer", desc_depth);

    ResourceDescription desc_entity;
    desc_entity.type           = ResourceType::Texture;
    desc_entity.texture.type   = type;
    desc_entity.texture.format = TextureFormat::RINT;

    uint32_t entity_handle = reflection.addOutput("entity_buffer", desc_entity);

    ResourceDescription desc_stencil;
    desc_stencil.type           = ResourceType::Texture;
    desc_stencil.texture.type   = type;
    desc_stencil.texture.format = TextureFormat::RINT8;

    uint32_t stencil_handle = reflection.addOutput("stencil_buffer", desc_stencil);

    reflection.setOutputFramebufferData(TextureSizeHint::FULL_FRAMEBUFFER,
                                        TextureSizeHint::FULL_FRAMEBUFFER,
                                        {color_handle, entity_handle, stencil_handle, depth_handle});

    return reflection;
}

void ForwardPass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto renderer    = ctx.renderer;
    auto scene       = ctx.scene;
    auto camera      = ctx.camera;
    const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

    atcg::ref_ptr<TextureCubeArray> point_light_depth_maps = resources.getTexture<TextureCubeArray>("point_light_depth_"
                                                                                                    "maps");
    atcg::ref_ptr<Texture2D> depth_buffer                  = resources.getTexture<Texture2D>("depth_buffer");

    bool has_skybox              = scene->hasSkybox();
    atcg::ref_ptr<Skybox> skybox = has_skybox ? scene->getSkybox() : AssetManager::getDummySkybox();

    Dictionary auxiliary;
    auxiliary.setValue("point_light_depth_maps", point_light_depth_maps);
    auxiliary.setValue("skybox", skybox);
    auxiliary.setValue("has_skybox", has_skybox);
    auxiliary.setValue("draw_cameras", ctx.draw_cameras);
    auxiliary.setValue("depth_map", depth_buffer);

    auto target = resources.getTargetFBO();

    GraphicsCommand::beginRenderPass(target);

    GraphicsCommand::clear();
    {
        int value = -1;
        target->getColorAttachement(1)->fill(&value);
    }

    {
        uint8_t value = 0;
        target->getColorAttachement(2)->fill(&value);
    }

    // Draw skybox
    if(has_skybox)
    {
        auto shader = renderer->getShaderManager()->getShader("skybox");
        auto cube   = AssetManager::getCubeMesh();
        GraphicsPipeline skybox_pipeline =
            GraphicsPipeline()
                .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                .setRasterizerState(RasterizerState().enableCulling(false).setDepthState(
                    DepthState().setDepthFunction(DepthFunction::ATCG_LEQUAL).enableDepthWrite(false)))
                .setShader(shader);

        uint32_t skybox_id = renderer->popTextureID();
        shader->setInt("skybox", skybox_id);
        GraphicsCommand::bindTexture(skybox_id, skybox->getSkyboxCubeMap());

        renderer->drawVAO(cube->getVerticesArray(), camera, glm::mat4(1), skybox_pipeline, cube->n_vertices());

        renderer->pushTextureID(skybox_id);
    }

    // Draw entities
    this->_transparent_entities.clear();
    for(auto e: view)
    {
        Entity entity(e, scene.get());
        if(entity.hasComponent<CustomRenderComponent>())
        {
            CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
            renderer.callback(entity, camera);
        }

        if(entity.hasAnyComponent<TransparencyComponent>() && entity.getComponent<TransparencyComponent>().transparent)
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
}

void ForwardPass::registerRenderPass(RenderPassRegistry::Registry* registry)
{
    ATCG_REGISTER_RENDER_PASS(registry, "ForwardPass", ForwardPass);
}

}    // namespace atcg