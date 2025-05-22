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

uint32_t setLights(const atcg::ref_ptr<atcg::Scene>& scene,
                   const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
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

        return shadow_map_id;
    }
    else
    {
        ATCG_ASSERT(num_lights == 0, "Shadow map is not initialized but lights are present");
    }

    return -1;
}

template<typename T>
void _renderComponent(const atcg::ref_ptr<atcg::Scene>& scene,
                      const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
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
        uint32_t id = setLights(scene, point_light_depth_maps, shader);
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             glm::vec3(1),
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE,
                             renderer.material,
                             entity.entity_handle());
        if(id != -1)
        {
            atcg::Renderer::pushTextureID(id);
        }
    }
}

template<>
void _renderComponent<PointRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                            const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
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
        uint32_t id = setLights(scene, point_light_depth_maps, shader);
        atcg::Renderer::setPointSize(renderer.point_size);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             renderer.color,
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_POINTS,
                             {},
                             entity.entity_handle());
        if(id != -1)
        {
            atcg::Renderer::pushTextureID(id);
        }
    }
}

template<>
void _renderComponent<PointSphereRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                                  const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
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
        uint32_t id = setLights(scene, point_light_depth_maps, shader);
        atcg::Renderer::setPointSize(renderer.point_size);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             glm::vec3(1),
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE,
                             renderer.material,
                             entity.entity_handle());
        if(id != -1)
        {
            atcg::Renderer::pushTextureID(id);
        }
    }
}

template<>
void _renderComponent<EdgeRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                           const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
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
        uint32_t id = setLights(scene, point_light_depth_maps, shader);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             renderer.color,
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_EDGES,
                             {},
                             entity.entity_handle());
        if(id != -1)
        {
            atcg::Renderer::pushTextureID(id);
        }
    }
}

template<>
void _renderComponent<EdgeCylinderRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                                   const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
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
        uint32_t id = setLights(scene, point_light_depth_maps, shader);
        shader->setFloat("edge_radius", renderer.radius);
        atcg::Renderer::draw(geometry.graph,
                             camera,
                             transform.getModel(),
                             glm::vec3(1),
                             shader,
                             atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER,
                             renderer.material,
                             entity.entity_handle());
        if(id != -1)
        {
            atcg::Renderer::pushTextureID(id);
        }
    }
}

template<>
void _renderComponent<InstanceRenderComponent>(const atcg::ref_ptr<atcg::Scene>& scene,
                                               const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
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

    _renderComponent<Component>(scene, point_light_depth_maps, entity, camera, geometry, transform, entity_id, shader);
}

void render(const atcg::ref_ptr<atcg::Scene>& scene,
            const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
            Entity entity,
            const atcg::ref_ptr<Camera>& camera)
{
    if(entity.hasComponent<CustomRenderComponent>())
    {
        CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
        renderer.callback(entity, camera);
    }

    renderComponent<MeshRenderComponent>(scene, point_light_depth_maps, entity, camera);
    renderComponent<PointRenderComponent>(scene, point_light_depth_maps, entity, camera);
    renderComponent<PointSphereRenderComponent>(scene, point_light_depth_maps, entity, camera);
    renderComponent<EdgeRenderComponent>(scene, point_light_depth_maps, entity, camera);
    renderComponent<EdgeCylinderRenderComponent>(scene, point_light_depth_maps, entity, camera);
    renderComponent<InstanceRenderComponent>(scene, point_light_depth_maps, entity, camera);
}
}    // namespace detail

class SceneRenderer::Impl
{
public:
    Impl();

    ~Impl();

    Dictionary context;

    atcg::ref_ptr<atcg::RenderGraph> graph;
};

SceneRenderer::Impl::Impl()
{
    graph = atcg::make_ref<atcg::RenderGraph>();

    auto [skybox_handle, skybox_builder] = graph->addRenderPass();    // Outputs into the current render target
    auto [shadow_handle, shadow_builder] = graph->addRenderPass();
    auto [output_handle, output_builder] = graph->addRenderPass();

    // SKYBOX PASS
    skybox_builder->setSetupFunction([](Dictionary&, Dictionary& data, Dictionary& output_data)
                                     { output_data.setValue("framebuffer", nullptr); });

    skybox_builder->setRenderFunction(
        [](Dictionary& context, const Dictionary&, Dictionary&, Dictionary&)
        { atcg::Renderer::drawSkybox(context.getValue<atcg::ref_ptr<Camera>>("camera")); });

    // SHADOW PASS
    shadow_builder->setSetupFunction(
        [](Dictionary&, Dictionary& data, Dictionary& output_data)
        {
            data.setValue("point_light_framebuffer", atcg::make_ref<atcg::Framebuffer>(1024, 1024));
            output_data.setValue("point_light_depth_maps",
                                 atcg::make_ref<atcg::ref_ptr<atcg::TextureCubeArray>>(nullptr));
        });

    shadow_builder->setRenderFunction(
        [](Dictionary& context, const Dictionary&, Dictionary& data, Dictionary& output_data)
        {
            auto scene = context.getValue<atcg::ref_ptr<Scene>>("scene");

            float n              = 0.1f;
            float f              = 100.0f;
            glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, n, f);

            const atcg::ref_ptr<Shader>& depth_pass_shader = atcg::ShaderManager::getShader("depth_pass");
            depth_pass_shader->setFloat("far_plane", f);

            uint32_t active_fbo = atcg::Framebuffer::currentFramebuffer();

            glm::vec4 old_viewport = atcg::Renderer::getViewport();

            auto light_view = scene->getAllEntitiesWith<PointLightComponent, TransformComponent>();

            uint32_t num_lights = 0;
            for(auto e: light_view)
            {
                ++num_lights;
            }

            auto point_light_depth_maps =
                output_data.getValue<atcg::ref_ptr<atcg::ref_ptr<atcg::TextureCubeArray>>>("point_light_"
                                                                                           "depth_maps");
            if(num_lights == 0)
            {
                *point_light_depth_maps = nullptr;
                return;
            }

            auto point_light_framebuffer = data.getValue<atcg::ref_ptr<atcg::Framebuffer>>("point_light_framebuffer");
            if(!(*point_light_depth_maps) || (*point_light_depth_maps)->depth() != num_lights)
            {
                atcg::TextureSpecification spec;
                spec.depth              = num_lights;
                spec.width              = 1024;
                spec.height             = 1024;
                spec.format             = atcg::TextureFormat::DEPTH;
                *point_light_depth_maps = atcg::TextureCubeArray::create(spec);

                point_light_framebuffer->attachDepth(*point_light_depth_maps);
                point_light_framebuffer->complete();
            }

            point_light_framebuffer->use();
            atcg::Renderer::setViewport(0, 0, point_light_framebuffer->width(), point_light_framebuffer->height());
            atcg::Renderer::clear();

            uint32_t light_idx = 0;
            for(auto e: light_view)
            {
                atcg::Entity entity(e, scene.get());

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

                const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent,
                                                             atcg::GeometryComponent,
                                                             atcg::MeshRenderComponent>();

                // Draw scene
                for(auto e: view)
                {
                    atcg::Entity entity(e, scene.get());

                    auto& transform = entity.getComponent<atcg::TransformComponent>();
                    auto& geometry  = entity.getComponent<atcg::GeometryComponent>();

                    atcg::Renderer::draw(geometry.graph,
                                         context.getValue<atcg::ref_ptr<Camera>>("camera"),
                                         transform.getModel(),
                                         glm::vec3(1),
                                         depth_pass_shader);
                }

                ++light_idx;
            }

            atcg::Renderer::setViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
            atcg::Framebuffer::bindByID(active_fbo);
        });

    output_builder->setRenderFunction(
        [](Dictionary& context, const Dictionary& inputs, Dictionary&, Dictionary&)
        {
            auto scene       = context.getValue<atcg::ref_ptr<Scene>>("scene");
            auto camera      = context.getValue<atcg::ref_ptr<Camera>>("camera");
            const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

            auto point_light_depth_maps = inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<atcg::TextureCubeArray>>>("point_"
                                                                                                                "light_"
                                                                                                                "depth_"
                                                                                                                "maps");
            for(auto e: view)
            {
                Entity entity(e, scene.get());
                // TODO
                detail::render(scene, *point_light_depth_maps, entity, camera);
            }
        });


    graph->addDependency(skybox_handle, "framebuffer", output_handle, "framebuffer");
    graph->addDependency(shadow_handle, "point_light_depth_maps", output_handle, "point_light_depth_maps");

    graph->compile(context);
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
    impl->context.setValue("scene", scene);
}

void SceneRenderer::render(const atcg::ref_ptr<Camera>& camera)
{
    impl->context.setValue("camera", camera);
    impl->graph->execute(impl->context);
}
}    // namespace atcg