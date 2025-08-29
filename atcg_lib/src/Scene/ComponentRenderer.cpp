#include <Scene/ComponentRenderer.h>

#include <Core/Assert.h>

namespace atcg
{

namespace detail
{

ATCG_INLINE atcg::ref_ptr<Skybox> getDummySkybox()
{
    static atcg::ref_ptr<Skybox> skybox = atcg::make_ref<Skybox>();
    return skybox;
}

ATCG_INLINE uint32_t setLights(atcg::RendererSystem* renderer,
                               Scene* scene,
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
        uint32_t shadow_map_id = renderer->popTextureID();
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

ATCG_INLINE std::pair<uint32_t, uint32_t>
setSkyLight(atcg::RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader, const atcg::ref_ptr<Skybox>& skybox)
{
    uint32_t irradiance_id = renderer->popTextureID();
    skybox->getIrradianceMap()->use(irradiance_id);
    shader->setInt("irradiance_map", irradiance_id);

    uint32_t prefiltered_id = renderer->popTextureID();
    skybox->getPrefilteredMap()->use(prefiltered_id);
    shader->setInt("prefilter_map", prefiltered_id);

    return std::make_pair(irradiance_id, prefiltered_id);
}
}    // namespace detail

void ComponentRenderer<MeshRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                             Entity entity,
                                                             const atcg::ref_ptr<Camera>& camera,
                                                             atcg::Dictionary& auxiliary) const
{
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

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    // Actual rendering of component
    MeshRenderComponent renderer = entity.getComponent<MeshRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", renderer.shader());

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", detail::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = detail::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = detail::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        _renderer->draw(geometry.graph(),
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE,
                        renderer.material(),
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

void ComponentRenderer<PointRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                              Entity entity,
                                                              const atcg::ref_ptr<Camera>& camera,
                                                              atcg::Dictionary& auxiliary) const
{
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

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    // Actual rendering of component
    PointRenderComponent renderer = entity.getComponent<PointRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", renderer.shader());

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", detail::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = detail::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = detail::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        _renderer->setPointSize(renderer.point_size);
        _renderer->draw(geometry.graph(),
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

void ComponentRenderer<PointSphereRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                                    Entity entity,
                                                                    const atcg::ref_ptr<Camera>& camera,
                                                                    atcg::Dictionary& auxiliary) const
{
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

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    // Actual rendering of component
    PointSphereRenderComponent renderer = entity.getComponent<PointSphereRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", renderer.shader());

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", detail::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = detail::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = detail::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        _renderer->setPointSize(renderer.point_size);
        _renderer->draw(geometry.graph(),
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE,
                        renderer.material(),
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

void ComponentRenderer<EdgeRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                             Entity entity,
                                                             const atcg::ref_ptr<Camera>& camera,
                                                             atcg::Dictionary& auxiliary) const
{
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

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    // Actual rendering of component
    EdgeRenderComponent renderer = entity.getComponent<EdgeRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader",
                                                    _renderer->getShaderManager()->getShader("edge"));

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", detail::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = detail::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = detail::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        _renderer->draw(geometry.graph(),
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

void ComponentRenderer<EdgeCylinderRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                                     Entity entity,
                                                                     const atcg::ref_ptr<Camera>& camera,
                                                                     atcg::Dictionary& auxiliary) const
{
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

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    // Actual rendering of component
    EdgeCylinderRenderComponent renderer = entity.getComponent<EdgeCylinderRenderComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader",
                                                    _renderer->getShaderManager()->getShader("cylinder_edge"));

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", detail::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    if(renderer.visible)
    {
        uint32_t id          = detail::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = detail::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setFloat("edge_radius", renderer.radius);
        _renderer->draw(geometry.graph(),
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER,
                        renderer.material(),
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

void ComponentRenderer<InstanceRenderComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                                 Entity entity,
                                                                 const atcg::ref_ptr<Camera>& camera,
                                                                 atcg::Dictionary& auxiliary) const
{
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

    if(!geometry.graph())
    {
        ATCG_WARN("Entity does have geometry component but mesh is empty");
        return;
    }

    geometry.graph()->unmapAllPointers();

    // Actual rendering of component
    InstanceRenderComponent renderer = entity.getComponent<InstanceRenderComponent>();

    auto override_shader =
        renderer.shader() ? renderer.shader() : _renderer->getShaderManager()->getShader("instanced");

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader", override_shader);

    auto point_light_depth_maps =
        auxiliary.getValueOr<atcg::ref_ptr<atcg::TextureCubeArray>>("point_light_depth_maps", nullptr);

    auto skybox     = auxiliary.getValueOr<atcg::ref_ptr<Skybox>>("skybox", detail::getDummySkybox());
    auto has_skybox = auxiliary.getValueOr<bool>("has_skybox", false);

    auto scene = entity.scene();

    if(renderer.visible)
    {
        auto vao = geometry.graph()->getVerticesArray();
        for(int i = 0; i < renderer.instance_vbos.size(); ++i)
        {
            renderer.instance_vbos[i]->unmapPointers();
            vao->pushInstanceBuffer(renderer.instance_vbos[i]);
        }

        uint32_t id          = detail::setLights(_renderer, scene, point_light_depth_maps, shader);
        auto [ir_id, pre_id] = detail::setSkyLight(_renderer, shader, skybox);
        shader->setInt("use_ibl", has_skybox);
        shader->setInt("receive_shadow", (int)renderer.receive_shadow);
        _renderer->draw(geometry.graph(),
                        camera,
                        transform.getModel(),
                        glm::vec3(1),
                        shader,
                        atcg::DrawMode::ATCG_DRAW_MODE_INSTANCED,
                        renderer.material(),
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