#include <Renderer/RenderPasses/VolumePass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
VolumePass::VolumePass(const RenderTargetDesc& desc) : RenderPass(desc, "VolumePass")
{
    registerOutput("framebuffer", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
    setSetupFunction(
        [this](Dictionary& context, Dictionary& data, Dictionary& output)
        {
            if(_render_target.mode == RenderTargetMode::RENDER_TARGET_OWN_FRAMEBUFFER)
            {
                data.setValue("target", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
            }

            data.setValue("front", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
            data.setValue("back", atcg::make_ref<atcg::ref_ptr<Framebuffer>>(nullptr));
        });

    setRenderFunction(
        [this](Dictionary& context, const Dictionary& inputs, Dictionary& data, Dictionary& outputs)
        {
            auto _renderer =
                context.getValueOr("renderer", atcg::SystemRegistry::instance()->getSystem<RendererSystem>());
            auto scene       = context.getValue<atcg::ref_ptr<Scene>>("scene");
            auto camera      = context.getValue<atcg::ref_ptr<Camera>>("camera");
            const auto& view = scene->getAllEntitiesWith<atcg::GeometryComponent,
                                                         atcg::TransformComponent,
                                                         atcg::MeshRenderComponent>();

            auto output_framebuffer = outputs.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("framebuffer");
            auto target             = prepareFramebuffer(context, inputs, data, outputs);
            *output_framebuffer     = target;

            // First render pass to determine ray directions and entry/exit points of the volume
            auto front = data.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("front");
            auto back  = data.getValue<atcg::ref_ptr<atcg::ref_ptr<Framebuffer>>>("back");

            if(!(*front) || (*front)->width() != target->width() || (*front)->height() != target->height())
            {
                *front = atcg::make_ref<Framebuffer>(target->width(), target->height());
                (*front)->attachDepth();
                (*front)->complete();
            }

            if(!(*back) || (*back)->width() != target->width() || (*back)->height() != target->height())
            {
                *back = atcg::make_ref<Framebuffer>(target->width(), target->height());
                (*back)->attachDepth();
                (*back)->complete();
            }

            auto volume_hom_shader          = _renderer->getShaderManager()->getShader("volume_hom");
            auto volume_het_shader          = _renderer->getShaderManager()->getShader("volume_het");
            auto depth_pass_shader          = _renderer->getShaderManager()->getShader("depth_pass_simple");
            GraphicsPipeline front_pipeline = GraphicsPipeline()
                                                  .setShader(depth_pass_shader)
                                                  .setRasterizerState(RasterizerState().enableCulling(true).setCullMode(
                                                      CullMode::ATCG_BACK_FACE_CULLING));
            GraphicsPipeline back_pipeline = GraphicsPipeline()
                                                 .setShader(depth_pass_shader)
                                                 .setRasterizerState(RasterizerState().enableCulling(true).setCullMode(
                                                     CullMode::ATCG_FRONT_FACE_CULLING));

            GraphicsPipeline volume_hom_pipeline =
                GraphicsPipeline()
                    .setShader(volume_hom_shader)
                    .setRasterizerState(
                        RasterizerState().enableCulling(true).setCullMode(CullMode::ATCG_BACK_FACE_CULLING));

            GraphicsPipeline volume_het_pipeline =
                GraphicsPipeline()
                    .setShader(volume_het_shader)
                    .setRasterizerState(
                        RasterizerState().enableCulling(true).setCullMode(CullMode::ATCG_BACK_FACE_CULLING));

            // Front pass
            GraphicsCommand::beginRenderPass(*front);
            GraphicsCommand::clear();
            for(auto e: view)
            {
                atcg::Entity entity(e, scene.get());

                if(!entity.hasAnyComponent<atcg::HomogeneousMediumComponent, atcg::HeterogeneousMediumComponent>())
                {
                    continue;
                }

                auto& renderer = entity.getComponent<MeshRenderComponent>();
                if(!renderer.visible || renderer.material()->getMaterialType() != MaterialType::MATERIAL_TYPE_NULL)
                {
                    continue;
                }

                auto& geometry = entity.getComponent<atcg::GeometryComponent>();
                if(!geometry.graph())
                {
                    continue;
                }

                _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                                   camera,
                                   entity.getComponent<atcg::TransformComponent>().getModel(),
                                   front_pipeline,
                                   geometry.graph()->n_vertices());
            }
            GraphicsCommand::endRenderPass();

            // Back pass
            GraphicsCommand::beginRenderPass(*back);
            GraphicsCommand::clear();
            for(auto e: view)
            {
                atcg::Entity entity(e, scene.get());

                if(!entity.hasAnyComponent<atcg::HomogeneousMediumComponent, atcg::HeterogeneousMediumComponent>())
                {
                    continue;
                }

                auto& renderer = entity.getComponent<MeshRenderComponent>();
                if(!renderer.visible || renderer.material()->getMaterialType() != MaterialType::MATERIAL_TYPE_NULL)
                {
                    continue;
                }

                auto& geometry = entity.getComponent<atcg::GeometryComponent>();
                if(!geometry.graph())
                {
                    continue;
                }

                _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                                   camera,
                                   entity.getComponent<atcg::TransformComponent>().getModel(),
                                   back_pipeline,
                                   geometry.graph()->n_vertices());
            }
            GraphicsCommand::endRenderPass();

            volume_hom_shader->setInt("front_depth", 0);
            volume_hom_shader->setInt("back_depth", 1);
            volume_het_shader->setInt("front_depth", 0);
            volume_het_shader->setInt("back_depth", 1);
            GraphicsCommand::bindTexture(0, (*front)->getDepthAttachement());
            GraphicsCommand::bindTexture(1, (*back)->getDepthAttachement());

            // Volume pass
            GraphicsCommand::beginRenderPass(target);
            for(auto e: view)
            {
                atcg::Entity entity(e, scene.get());

                if(!entity.hasAnyComponent<atcg::HomogeneousMediumComponent, atcg::HeterogeneousMediumComponent>())
                {
                    continue;
                }

                auto& renderer = entity.getComponent<MeshRenderComponent>();
                if(!renderer.visible || renderer.material()->getMaterialType() != MaterialType::MATERIAL_TYPE_NULL)
                {
                    continue;
                }

                auto& geometry = entity.getComponent<atcg::GeometryComponent>();
                if(!geometry.graph())
                {
                    continue;
                }

                if(entity.hasComponent<atcg::HomogeneousMediumComponent>())
                {
                    auto& medium = entity.getComponent<atcg::HomogeneousMediumComponent>();
                    volume_hom_shader->setVec3("albedo", medium.albedo);
                    volume_hom_shader->setFloat("density", medium.density);
                    volume_hom_shader->setInt("entityID", entity.entity_handle());
                    volume_hom_shader->setMat4("invView", glm::inverse(camera->getView()));
                    volume_hom_shader->setMat4("invProj", glm::inverse(camera->getProjection()));
                    volume_hom_shader->setFloat("g", medium.g);
                    volume_hom_shader->setFloat("Le", medium.Le);
                    volume_hom_shader->setVec3("Le_color", medium.Le_color);

                    _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                                       camera,
                                       entity.getComponent<atcg::TransformComponent>().getModel(),
                                       volume_hom_pipeline,
                                       geometry.graph()->n_vertices());
                }
                else if(entity.hasComponent<atcg::HeterogeneousMediumComponent>())
                {
                    auto& medium = entity.getComponent<atcg::HeterogeneousMediumComponent>();
                    volume_het_shader->setInt("entityID", entity.entity_handle());
                    volume_het_shader->setMat4("invView", glm::inverse(camera->getView()));
                    volume_het_shader->setMat4("invProj", glm::inverse(camera->getProjection()));
                    volume_het_shader->setFloat("g", medium.g);

                    // Bind density, albedo and emission textures
                    if(medium.density())
                    {
                        volume_het_shader->setInt("density_grid", 2);
                        volume_het_shader->setFloat("density_scale", medium.density_grid.scale);
                        glm::mat4 to_uvw = glm::mat4(1);
                        glm::vec3 scale  = medium.density_grid.bbox.max - medium.density_grid.bbox.min;
                        to_uvw           = to_uvw * glm::scale(1.0f / scale);
                        to_uvw           = to_uvw * glm::translate(-medium.density_grid.bbox.min);
                        to_uvw = to_uvw * glm::inverse(entity.getComponent<atcg::TransformComponent>().getModel());
                        volume_het_shader->setMat4("density_to_uvw", to_uvw);
                        GraphicsCommand::bindTexture(2, medium.density());
                    }

                    if(medium.albedo())
                    {
                        volume_het_shader->setInt("albedo_grid", 3);
                        volume_het_shader->setFloat("albedo_scale", medium.albedo_grid.scale);
                        glm::mat4 to_uvw = glm::mat4(1);
                        glm::vec3 scale  = medium.albedo_grid.bbox.max - medium.albedo_grid.bbox.min;
                        to_uvw           = to_uvw * glm::scale(1.0f / scale);
                        to_uvw           = to_uvw * glm::translate(-medium.albedo_grid.bbox.min);
                        to_uvw = to_uvw * glm::inverse(entity.getComponent<atcg::TransformComponent>().getModel());
                        volume_het_shader->setMat4("albedo_to_uvw", to_uvw);
                        GraphicsCommand::bindTexture(3, medium.albedo());
                    }

                    if(medium.emission())
                    {
                        volume_het_shader->setInt("emission_grid", 4);
                        volume_het_shader->setFloat("emission_scale", medium.emission_grid.scale);
                        glm::mat4 to_uvw = glm::mat4(1);
                        glm::vec3 scale  = medium.emission_grid.bbox.max - medium.emission_grid.bbox.min;
                        to_uvw           = to_uvw * glm::scale(1.0f / scale);
                        to_uvw           = to_uvw * glm::translate(-medium.emission_grid.bbox.min);
                        to_uvw = to_uvw * glm::inverse(entity.getComponent<atcg::TransformComponent>().getModel());
                        volume_het_shader->setMat4("emission_to_uvw", to_uvw);
                        GraphicsCommand::bindTexture(4, medium.emission());
                    }

                    _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                                       camera,
                                       entity.getComponent<atcg::TransformComponent>().getModel(),
                                       volume_het_pipeline,
                                       geometry.graph()->n_vertices());
                }
            }
            GraphicsCommand::endRenderPass();
        });
}
}    // namespace atcg