#include <Scene/ComponentRenderer.h>

#include <Core/Assert.h>

namespace atcg
{

uint32_t ComponentRenderer::_setLights(Scene* scene,
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

        atcg::Entity light_entity(e, scene);

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
        shader->setInt("shadow_pass", 1);
        point_light_depth_maps->use(shadow_map_id);

        return shadow_map_id;
    }
    else
    {
        shader->setInt("shadow_pass", 0);
        //     ATCG_ASSERT(num_lights == 0, "Shadow map is not initialized but lights are present");
    }

    return -1;
}

template<typename T>
void ComponentRenderer::renderComponent(Entity entity, const atcg::ref_ptr<Camera>& camera, atcg::Dictionary& auxiliary)
{
}

template<>
void ComponentRenderer::renderComponent<MeshRenderComponent>(Entity entity,
                                                             const atcg::ref_ptr<Camera>& camera,
                                                             atcg::Dictionary& auxiliary)
{
    if(!entity.hasComponent<MeshRenderComponent>()) return;

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

    // Actual rendering of component
    MeshRenderComponent renderer = entity.getComponent<MeshRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> override_shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", nullptr);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        uint32_t id = _setLights(scene, point_light_depth_maps, shader);
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
void ComponentRenderer::renderComponent<PointRenderComponent>(Entity entity,
                                                              const atcg::ref_ptr<Camera>& camera,
                                                              atcg::Dictionary& auxiliary)
{
    if(!entity.hasComponent<PointRenderComponent>()) return;

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

    // Actual rendering of component
    PointRenderComponent renderer = entity.getComponent<PointRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> override_shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", nullptr);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        uint32_t id = _setLights(scene, point_light_depth_maps, shader);
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
void ComponentRenderer::renderComponent<PointSphereRenderComponent>(Entity entity,
                                                                    const atcg::ref_ptr<Camera>& camera,
                                                                    atcg::Dictionary& auxiliary)
{
    if(!entity.hasComponent<PointSphereRenderComponent>()) return;

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

    // Actual rendering of component
    PointSphereRenderComponent renderer = entity.getComponent<PointSphereRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> override_shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", nullptr);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto shader = override_shader ? override_shader : renderer.shader;

    if(renderer.visible)
    {
        uint32_t id = _setLights(scene, point_light_depth_maps, shader);
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
void ComponentRenderer::renderComponent<EdgeRenderComponent>(Entity entity,
                                                             const atcg::ref_ptr<Camera>& camera,
                                                             atcg::Dictionary& auxiliary)
{
    if(!entity.hasComponent<EdgeRenderComponent>()) return;

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

    // Actual rendering of component
    EdgeRenderComponent renderer = entity.getComponent<EdgeRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> override_shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", nullptr);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto shader = override_shader ? override_shader : atcg::ShaderManager::getShader("edge");

    if(renderer.visible)
    {
        uint32_t id = _setLights(scene, point_light_depth_maps, shader);
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
void ComponentRenderer::renderComponent<EdgeCylinderRenderComponent>(Entity entity,
                                                                     const atcg::ref_ptr<Camera>& camera,
                                                                     atcg::Dictionary& auxiliary)
{
    if(!entity.hasComponent<EdgeCylinderRenderComponent>()) return;

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

    // Actual rendering of component
    EdgeCylinderRenderComponent renderer = entity.getComponent<EdgeCylinderRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> override_shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", nullptr);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto shader = override_shader ? override_shader : atcg::ShaderManager::getShader("cylinder_edge");

    if(renderer.visible)
    {
        uint32_t id = _setLights(scene, point_light_depth_maps, shader);
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
void ComponentRenderer::renderComponent<InstanceRenderComponent>(Entity entity,
                                                                 const atcg::ref_ptr<Camera>& camera,
                                                                 atcg::Dictionary& auxiliary)
{
    if(!entity.hasComponent<InstanceRenderComponent>()) return;

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

    // Actual rendering of component
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
}    // namespace atcg