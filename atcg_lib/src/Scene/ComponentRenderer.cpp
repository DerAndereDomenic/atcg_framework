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
        uint32_t shadow_map_id = _renderer->popTextureID();
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

std::pair<uint32_t, uint32_t> ComponentRenderer::_setSkyLight(const atcg::ref_ptr<Shader>& shader,
                                                              const atcg::ref_ptr<Skybox>& skybox)
{
    uint32_t irradiance_id = _renderer->popTextureID();
    skybox->getIrradianceMap()->use(irradiance_id);
    shader->setInt("irradiance_map", irradiance_id);

    uint32_t prefiltered_id = _renderer->popTextureID();
    skybox->getPrefilteredMap()->use(prefiltered_id);
    shader->setInt("prefilter_map", prefiltered_id);

    return std::make_pair(irradiance_id, prefiltered_id);
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

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", renderer.shader);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", _dummy_skybox);
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = _setLights(scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = _setSkyLight(shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        _renderer->draw(geometry.graph,
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE,
                        renderer.material,
                        entity.entity_handle());
        if(id != -1)
        {
            _renderer->pushTextureID(id);
        }
        if(ir_id != -1)
        {
            _renderer->pushTextureID(ir_id);
        }
        if(pre_id != -1)
        {
            _renderer->pushTextureID(pre_id);
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

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", renderer.shader);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", _dummy_skybox);
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = _setLights(scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = _setSkyLight(shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        _renderer->setPointSize(renderer.point_size);
        _renderer->draw(geometry.graph,
                        camera,
                        transform.getModel(),
                        renderer.color,
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_POINTS,
                        {},
                        entity.entity_handle());
        if(id != -1)
        {
            _renderer->pushTextureID(id);
        }
        if(ir_id != -1)
        {
            _renderer->pushTextureID(ir_id);
        }
        if(pre_id != -1)
        {
            _renderer->pushTextureID(pre_id);
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

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", renderer.shader);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", _dummy_skybox);
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = _setLights(scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = _setSkyLight(shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        _renderer->setPointSize(renderer.point_size);
        _renderer->draw(geometry.graph,
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE,
                        renderer.material,
                        entity.entity_handle());
        if(id != -1)
        {
            _renderer->pushTextureID(id);
        }
        if(ir_id != -1)
        {
            _renderer->pushTextureID(ir_id);
        }
        if(pre_id != -1)
        {
            _renderer->pushTextureID(pre_id);
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

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", atcg::ShaderManager::getShader("edge"));

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", _dummy_skybox);
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = _setLights(scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = _setSkyLight(shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        _renderer->draw(geometry.graph,
                        camera,
                        transform.getModel(),
                        renderer.color,
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_EDGES,
                        {},
                        entity.entity_handle());
        if(id != -1)
        {
            _renderer->pushTextureID(id);
        }
        if(ir_id != -1)
        {
            _renderer->pushTextureID(ir_id);
        }
        if(pre_id != -1)
        {
            _renderer->pushTextureID(pre_id);
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

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", atcg::ShaderManager::getShader("cylinder_edge"));

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", _dummy_skybox);
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = _setLights(scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = _setSkyLight(shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setFloat("edge_radius", renderer.radius);
        _renderer->draw(geometry.graph,
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER,
                        renderer.material,
                        entity.entity_handle());
        if(id != -1)
        {
            _renderer->pushTextureID(id);
        }
        if(ir_id != -1)
        {
            _renderer->pushTextureID(ir_id);
        }
        if(pre_id != -1)
        {
            _renderer->pushTextureID(pre_id);
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
    InstanceRenderComponent renderer = entity.getComponent<InstanceRenderComponent>();

    auto override_shader = renderer.shader ? renderer.shader : atcg::ShaderManager::getShader("instanced");

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", override_shader);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", _dummy_skybox);
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    auto scene = entity.scene();

    if(renderer.visible)
    {
        auto vao = geometry.graph->getVerticesArray();
        for(int i = 0; i < renderer.instance_vbos.size(); ++i)
        {
            renderer.instance_vbos[i]->unmapPointers();
            vao->pushInstanceBuffer(renderer.instance_vbos[i]);
        }

        uint32_t id          = _setLights(scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = _setSkyLight(shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        // shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        _renderer->draw(geometry.graph,
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_INSTANCED,
                        renderer.material,
                        entity.entity_handle());
        if(id != -1)
        {
            _renderer->pushTextureID(id);
        }
        if(ir_id != -1)
        {
            _renderer->pushTextureID(ir_id);
        }
        if(pre_id != -1)
        {
            _renderer->pushTextureID(pre_id);
        }

        for(int i = 0; i < renderer.instance_vbos.size(); ++i)
        {
            vao->popVertexBuffer();
        }
    }
}
}    // namespace atcg