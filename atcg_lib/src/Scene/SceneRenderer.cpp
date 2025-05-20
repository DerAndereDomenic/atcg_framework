#include <Scene/SceneRenderer.h>

#include <Core/Assert.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Renderer/RenderGraph.h>

namespace atcg
{

namespace detail
{

void freeTextureUnits(std::vector<uint32_t>& used_texture_units)
{
    for(uint32_t texture_id: used_texture_units)
    {
        atcg::Renderer::pushTextureID(texture_id);
    }
    used_texture_units.clear();
}

void setLights(const atcg::ref_ptr<atcg::Scene>& scene,
               const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
               std::vector<uint32_t>& used_texture_units,
               const atcg::ref_ptr<Shader>& shader)
{
    auto light_view = scene->getAllEntitiesWith<atcg::PointLightComponent, atcg::TransformComponent>();

    uint32_t num_lights = 0;
    for(auto e: light_view)
    {
        std::stringstream light_index;
        light_index << "[" << num_lights << "]";
        std::string light_index_str = light_index.str();

        atcg::Entity light_entity(e, scene.get());

        auto& point_light     = light_entity.getComponent<atcg::PointLightComponent>();
        auto& light_transform = light_entity.getComponent<atcg::TransformComponent>();

        shader->setVec3("light_colors" + light_index_str, point_light.color);
        shader->setFloat("light_intensities" + light_index_str, point_light.intensity);
        shader->setVec3("light_positions" + light_index_str, light_transform.getPosition());

        ++num_lights;
    }

    shader->setInt("num_lights", num_lights);
    if(point_light_depth_maps)
    {
        uint32_t shadow_map_id = atcg::Renderer::popTextureID();
        shader->setInt("shadow_maps", shadow_map_id);
        point_light_depth_maps->use(shadow_map_id);

        used_texture_units.push_back(shadow_map_id);
    }
    else
    {
        ATCG_ASSERT(num_lights == 0, "Shadow map is not initialized but lights are present");
    }
}

template<typename T>
void _renderComponent(const atcg::ref_ptr<atcg::Scene>& scene,
                      const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                      std::vector<uint32_t>& used_texture_units,
                      Entity entity,
                      const atcg::ref_ptr<Camera>& camera,
                      const GeometryComponent& geometry,
                      const TransformComponent& transform,
                      const uint32_t entity_id,
                      const atcg::ref_ptr<atcg::Shader>& override_shader)
{
}

template<>
void _renderComponent<MeshRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                           const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                                           std::vector<uint32_t>& used_texture_units,
                                           Entity entity,
                                           const atcg::ref_ptr<Camera>& camera,
                                           const GeometryComponent& geometry,
                                           const TransformComponent& transform,
                                           const uint32_t entity_id,
                                           const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    MeshRenderComponent renderer = entity.getComponent<MeshRenderComponent>();

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        setLights(scene, point_light_depth_maps, used_texture_units, shader);
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             glm::vec3(1),
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE,
                             renderer.material,
                             entity.entity_handle());
        freeTextureUnits(used_texture_units);
    }
}

template<>
void _renderComponent<PointRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                            const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                                            std::vector<uint32_t>& used_texture_units,
                                            Entity entity,
                                            const atcg::ref_ptr<Camera>& camera,
                                            const GeometryComponent& geometry,
                                            const TransformComponent& transform,
                                            const uint32_t entity_id,
                                            const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    PointRenderComponent renderer = entity.getComponent<PointRenderComponent>();

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        setLights(scene, point_light_depth_maps, used_texture_units, shader);
        atcg::Renderer::setPointSize(renderer.point_size);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             renderer.color,
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_POINTS,
                             {},
                             entity.entity_handle());
        freeTextureUnits(used_texture_units);
    }
}

template<>
void _renderComponent<PointSphereRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                                  const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                                                  std::vector<uint32_t>& used_texture_units,
                                                  Entity entity,
                                                  const atcg::ref_ptr<Camera>& camera,
                                                  const GeometryComponent& geometry,
                                                  const TransformComponent& transform,
                                                  const uint32_t entity_id,
                                                  const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    PointSphereRenderComponent renderer = entity.getComponent<PointSphereRenderComponent>();

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        setLights(scene, point_light_depth_maps, used_texture_units, shader);
        atcg::Renderer::setPointSize(renderer.point_size);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             glm::vec3(1),
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE,
                             renderer.material,
                             entity.entity_handle());
        freeTextureUnits(used_texture_units);
    }
}

template<>
void _renderComponent<EdgeRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                           const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                                           std::vector<uint32_t>& used_texture_units,
                                           Entity entity,
                                           const atcg::ref_ptr<Camera>& camera,
                                           const GeometryComponent& geometry,
                                           const TransformComponent& transform,
                                           const uint32_t entity_id,
                                           const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    EdgeRenderComponent renderer = entity.getComponent<EdgeRenderComponent>();

    auto shader = override_shader ? override_shader : atcg::ShaderManager::getShader("edge");

    if(renderer.visible)
    {
        setLights(scene, point_light_depth_maps, used_texture_units, shader);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             renderer.color,
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_EDGES,
                             {},
                             entity.entity_handle());
        freeTextureUnits(used_texture_units);
    }
}

template<>
void _renderComponent<EdgeCylinderRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                                   const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                                                   std::vector<uint32_t>& used_texture_units,
                                                   Entity entity,
                                                   const atcg::ref_ptr<Camera>& camera,
                                                   const GeometryComponent& geometry,
                                                   const TransformComponent& transform,
                                                   const uint32_t entity_id,
                                                   const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    EdgeCylinderRenderComponent renderer = entity.getComponent<EdgeCylinderRenderComponent>();

    auto shader = override_shader ? override_shader : atcg::ShaderManager::getShader("cylinder_edge");

    if(renderer.visible)
    {
        setLights(scene, point_light_depth_maps, used_texture_units, shader);
        shader->setFloat("edge_radius", renderer.radius);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             glm::vec3(1),
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER,
                             renderer.material,
                             entity.entity_handle());
        freeTextureUnits(used_texture_units);
    }
}

template<>
void _renderComponent<InstanceRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                               const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                                               std::vector<uint32_t>& used_texture_units,
                                               Entity entity,
                                               const atcg::ref_ptr<Camera>& camera,
                                               const GeometryComponent& geometry,
                                               const TransformComponent& transform,
                                               const uint32_t entity_id,
                                               const atcg::ref_ptr<atcg::Shader>& override_shader)
{
    // TODO
    //  InstanceRenderComponent renderer = entity.getComponent<InstanceRenderComponent>();

    // auto shader = override_shader ? override_shader : shader_manager->getShader("instanced");

    // if(renderer.visible)
    // {
    //     if(geometry.graph->getVerticesArray()->peekVertexBuffer() != renderer.instance_vbo)
    //     {
    //         geometry.graph->getVerticesArray()->pushInstanceBuffer(renderer.instance_vbo);
    //     }

    //     shader->setInt("entityID", entity_id);
    //     setMaterial(renderer.material, shader);
    //     atcg::ref_ptr<VertexArray> vao_mesh      = geometry.graph->getVerticesArray();
    //     atcg::ref_ptr<VertexBuffer> instance_vbo = vao_mesh->peekVertexBuffer();
    //     uint32_t n_instances                     = instance_vbo->size() / instance_vbo->getLayout().getStride();
    //     drawVAO(vao_mesh,
    //             camera,
    //             glm::vec3(1),
    //             shader,
    //             transform.getModel(),
    //             GL_TRIANGLES,
    //             geometry.graph->n_vertices(),
    //             n_instances);
    //     freeTextureUnits();
    // }
}

template<typename Component>
void renderComponent(const atcg::ref_ptr<atcg::Scene>& scene,
                     const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                     std::vector<uint32_t>& used_texture_units,
                     Entity entity,
                     const atcg::ref_ptr<Camera>& camera,
                     const atcg::ref_ptr<atcg::Shader>& shader = nullptr)
{
    if(!entity.hasComponent<Component>()) return;

    if(!entity.hasComponent<TransformComponent>())
    {
        ATCG_WARN("Entity does not have transform component!");
        return;
    }

    if(!entity.hasComponent<GeometryComponent>())
    {
        ATCG_WARN("Entity does not have geometry component!");
        return;
    }

    uint32_t entity_id           = entity.entity_handle();
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    if(!geometry.graph)
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph->unmapAllPointers();

    _renderComponent<Component>(scene,
                                point_light_depth_maps,
                                used_texture_units,
                                entity,
                                camera,
                                geometry,
                                transform,
                                entity_id,
                                shader);
}

void render(const atcg::ref_ptr<atcg::Scene>& scene,
            const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
            std::vector<uint32_t>& used_texture_units,
            Entity entity,
            const atcg::ref_ptr<Camera>& camera)
{
    if(entity.hasComponent<CustomRenderComponent>())
    {
        CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
        renderer.callback(entity, camera);
    }

    renderComponent<MeshRenderComponent>(scene, point_light_depth_maps, used_texture_units, entity, camera);
    renderComponent<PointRenderComponent>(scene, point_light_depth_maps, used_texture_units, entity, camera);
    renderComponent<PointSphereRenderComponent>(scene, point_light_depth_maps, used_texture_units, entity, camera);
    renderComponent<EdgeRenderComponent>(scene, point_light_depth_maps, used_texture_units, entity, camera);
    renderComponent<EdgeCylinderRenderComponent>(scene, point_light_depth_maps, used_texture_units, entity, camera);
    renderComponent<InstanceRenderComponent>(scene, point_light_depth_maps, used_texture_units, entity, camera);
}
}    // namespace detail

class SceneRenderer::Impl
{
public:
    Impl();

    ~Impl();

    struct RenderContext
    {
        atcg::ref_ptr<atcg::Scene> scene;
        atcg::ref_ptr<atcg::Camera> camera;
    };

    struct ShadowMappingData
    {
        atcg::ref_ptr<atcg::Framebuffer> point_light_framebuffer;
        atcg::ref_ptr<atcg::TextureCubeArray> point_light_depth_maps;
    };

    atcg::ref_ptr<RenderContext> context;

    atcg::ref_ptr<atcg::RenderGraph<RenderContext>> graph;
};

SceneRenderer::Impl::Impl()
{
    context = atcg::make_ref<RenderContext>();

    graph = atcg::make_ref<atcg::RenderGraph<RenderContext>>(context);

    auto [skybox_handle, skybox_builder] =
        graph->addRenderPass<int, int>();    // Outputs into the current render target
    auto [shadow_handle, shadow_builder] = graph->addRenderPass<int, ShadowMappingData>();
    auto [output_handle, output_builder] = graph->addRenderPass<int, int>();

    // SKYBOX PASS
    skybox_builder->setRenderFunction([](const atcg::ref_ptr<RenderContext>& context,
                                         const std::vector<std::any>&,
                                         const atcg::ref_ptr<int>&,
                                         const atcg::ref_ptr<int>&) { atcg::Renderer::drawSkybox(context->camera); });

    // SHADOW PASS
    shadow_builder->setSetupFunction(
        [](const atcg::ref_ptr<RenderContext>&,
           const atcg::ref_ptr<int>&,
           atcg::ref_ptr<ShadowMappingData>& output_data)
        { output_data->point_light_framebuffer = atcg::make_ref<atcg::Framebuffer>(1024, 1024); });

    shadow_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>&,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<ShadowMappingData>& output_data)
        {
            float n              = 0.1f;
            float f              = 100.0f;
            glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, n, f);

            const atcg::ref_ptr<Shader>& depth_pass_shader = atcg::ShaderManager::getShader("depth_pass");
            depth_pass_shader->setFloat("far_plane", f);

            uint32_t active_fbo = atcg::Framebuffer::currentFramebuffer();

            glm::vec4 old_viewport = atcg::Renderer::getViewport();

            auto light_view = context->scene->getAllEntitiesWith<PointLightComponent, TransformComponent>();

            uint32_t num_lights = 0;
            for(auto e: light_view)
            {
                ++num_lights;
            }

            if(num_lights == 0)
            {
                output_data->point_light_depth_maps = nullptr;
                return;
            }

            if(!output_data->point_light_depth_maps || output_data->point_light_depth_maps->depth() != num_lights)
            {
                atcg::TextureSpecification spec;
                spec.depth                          = num_lights;
                spec.width                          = 1024;
                spec.height                         = 1024;
                spec.format                         = atcg::TextureFormat::DEPTH;
                output_data->point_light_depth_maps = atcg::TextureCubeArray::create(spec);

                output_data->point_light_framebuffer->attachDepth(output_data->point_light_depth_maps);
                output_data->point_light_framebuffer->complete();
            }

            output_data->point_light_framebuffer->use();
            atcg::Renderer::setViewport(0,
                                        0,
                                        output_data->point_light_framebuffer->width(),
                                        output_data->point_light_framebuffer->height());
            atcg::Renderer::clear();

            uint32_t light_idx = 0;
            for(auto e: light_view)
            {
                atcg::Entity entity(e, context->scene.get());

                auto& point_light = entity.getComponent<PointLightComponent>();
                auto& transform   = entity.getComponent<TransformComponent>();

                if(!point_light.cast_shadow)
                {
                    ++light_idx;
                    continue;
                }

                glm::vec3 lightPos = transform.getPosition();
                depth_pass_shader->setVec3("lightPos", lightPos);
                depth_pass_shader->setMat4(
                    "shadowMatrices[0]",
                    projection * glm::lookAt(lightPos, lightPos + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
                depth_pass_shader->setMat4("shadowMatrices[1]",
                                           projection * glm::lookAt(lightPos,
                                                                    lightPos + glm::vec3(-1.0, 0.0, 0.0),
                                                                    glm::vec3(0.0, -1.0, 0.0)));
                depth_pass_shader->setMat4(
                    "shadowMatrices[2]",
                    projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
                depth_pass_shader->setMat4("shadowMatrices[3]",
                                           projection * glm::lookAt(lightPos,
                                                                    lightPos + glm::vec3(0.0, -1.0, 0.0),
                                                                    glm::vec3(0.0, 0.0, -1.0)));
                depth_pass_shader->setMat4(
                    "shadowMatrices[4]",
                    projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
                depth_pass_shader->setMat4("shadowMatrices[5]",
                                           projection * glm::lookAt(lightPos,
                                                                    lightPos + glm::vec3(0.0, 0.0, -1.0),
                                                                    glm::vec3(0.0, -1.0, 0.0)));
                depth_pass_shader->setInt("light_idx", light_idx);

                const auto& view = context->scene->getAllEntitiesWith<atcg::TransformComponent,
                                                                      atcg::GeometryComponent,
                                                                      atcg::MeshRenderComponent>();

                // Draw scene
                for(auto e: view)
                {
                    atcg::Entity entity(e, context->scene.get());

                    auto& transform = entity.getComponent<atcg::TransformComponent>();
                    auto& geometry  = entity.getComponent<atcg::GeometryComponent>();

                    atcg::Renderer::draw(geometry.graph,
                                         context->camera,
                                         transform.getModel(),
                                         glm::vec3(1),
                                         depth_pass_shader);
                }

                ++light_idx;
            }

            atcg::Renderer::setViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
            atcg::Framebuffer::bindByID(active_fbo);
        });

    output_builder->setSetupFunction(
        [](const atcg::ref_ptr<RenderContext>&, const atcg::ref_ptr<int>&, atcg::ref_ptr<int>&) {});

    output_builder->setRenderFunction(
        [](const atcg::ref_ptr<RenderContext>& context,
           const std::vector<std::any>& inputs,
           const atcg::ref_ptr<int>&,
           const atcg::ref_ptr<int>&)
        {
            const auto& view = context->scene->getAllEntitiesWith<atcg::TransformComponent>();

            auto shadow_maps = std::any_cast<atcg::ref_ptr<ShadowMappingData>>(inputs[1]);
            std::vector<uint32_t> used_texture_units;    // TODO
            for(auto e: view)
            {
                Entity entity(e, context->scene.get());
                // TODO
                detail::render(context->scene,
                               shadow_maps->point_light_depth_maps,
                               used_texture_units,
                               entity,
                               context->camera);
            }
        });


    graph->addDependency(skybox_handle, output_handle);
    graph->addDependency(shadow_handle, output_handle);

    graph->compile();
}

SceneRenderer::Impl::~Impl() {}

SceneRenderer::SceneRenderer()
{
    impl = std::make_unique<Impl>();
}

SceneRenderer::SceneRenderer(const atcg::ref_ptr<atcg::Scene>& scene)
{
    impl = std::make_unique<Impl>();
    setScene(scene);
}

SceneRenderer::~SceneRenderer() {}

void SceneRenderer::setScene(const atcg::ref_ptr<atcg::Scene>& scene)
{
    impl->context->scene = scene;
}

void SceneRenderer::render(const atcg::ref_ptr<Camera>& camera)
{
    ATCG_ASSERT(impl->context->scene, "No scene set");

    impl->context->camera = camera;
    impl->graph->execute();
}
}    // namespace atcg