#include <Scene/ComponentRenderer.h>

#include <Core/Assert.h>

#include <glad/glad.h>

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
        renderer->bindTexture(shadow_map_id, point_light_depth_maps);

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
    renderer->bindTexture(irradiance_id, skybox->getIrradianceMap());
    shader->setInt("irradiance_map", irradiance_id);

    uint32_t prefiltered_id = renderer->popTextureID();
    renderer->bindTexture(prefiltered_id, skybox->getPrefilteredMap());
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
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        _renderer->bindTexture(lut_id, AssetManager::getLUTTexture());

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);

        _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_vertices());
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
        if(lut_id != -1)
        {
            _renderer->pushTextureID(lut_id);
        }
        renderer.material()->releaseTextureIDs(_renderer);
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
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", renderer.color);
        renderer.default_material->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        _renderer->bindTexture(lut_id, AssetManager::getLUTTexture());

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setRasterizerState(RasterizerState().setPointSize(renderer.point_size))
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS);

        _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_vertices());
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
        if(lut_id != -1)
        {
            _renderer->pushTextureID(lut_id);
        }
        renderer.default_material->releaseTextureIDs(_renderer);
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
        shader->setInt("entityID", entity.entity_handle());
        shader->setFloat("point_size", renderer.point_size);
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        _renderer->bindTexture(lut_id, AssetManager::getLUTTexture());

        auto vbo = geometry.graph()->getVerticesBuffer();

        auto sphere_mesh = AssetManager::getSphereMesh();
        auto vao_sphere  = sphere_mesh->getVerticesArray();

        vao_sphere->pushInstanceBuffer(vbo);

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);

        // _renderer->setPointSize(renderer.point_size);
        _renderer->drawVAO(vao_sphere,
                           camera,
                           transform.getModel(),
                           pipeline,
                           sphere_mesh->n_vertices(),
                           geometry.graph()->n_vertices());
        vao_sphere->popVertexBuffer();
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
        if(lut_id != -1)
        {
            _renderer->pushTextureID(lut_id);
        }
        renderer.material()->releaseTextureIDs(_renderer);
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
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", renderer.color);
        renderer.default_material->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        _renderer->bindTexture(lut_id, AssetManager::getLUTTexture());

        auto points = geometry.graph()->getVerticesBuffer();
        _renderer->bindStorageBuffer(0, points);

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setRasterizerState(RasterizerState().setLineSize(1.0f))    // TODO
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS);

        _renderer->drawVAO(geometry.graph()->getEdgesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_edges());
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
        if(lut_id != -1)
        {
            _renderer->pushTextureID(lut_id);
        }
        renderer.default_material->releaseTextureIDs(_renderer);
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
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        _renderer->bindTexture(lut_id, AssetManager::getLUTTexture());

        auto points  = geometry.graph()->getVerticesBuffer();
        auto indices = geometry.graph()->getEdgesBuffer();

        _renderer->bindStorageBuffer(0, points);

        auto cylinder_mesh = AssetManager::getCylinderMesh();
        auto vao_cylinder  = cylinder_mesh->getVerticesArray();

        vao_cylinder->pushInstanceBuffer(indices);

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);

        _renderer->drawVAO(vao_cylinder,
                           camera,
                           transform.getModel(),
                           pipeline,
                           cylinder_mesh->n_vertices(),
                           geometry.graph()->n_edges());
        vao_cylinder->popVertexBuffer();
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
        if(lut_id != -1)
        {
            _renderer->pushTextureID(lut_id);
        }
        renderer.material()->releaseTextureIDs(_renderer);
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
        shader->setInt("entityID", entity.entity_handle());
        shader->setVec3("flat_color", glm::vec3(1));
        renderer.material()->uploadMaterial(_renderer, shader);
        uint32_t lut_id = _renderer->popTextureID();
        shader->setInt("lut", lut_id);
        _renderer->bindTexture(lut_id, AssetManager::getLUTTexture());

        auto instance_vbo    = vao->peekVertexBuffer();
        uint32_t n_instances = instance_vbo->size() / instance_vbo->getLayout().getStride();

        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);

        _renderer->drawVAO(vao, camera, transform.getModel(), pipeline, geometry.graph()->n_vertices(), n_instances);
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
        if(lut_id != -1)
        {
            _renderer->pushTextureID(lut_id);
        }
        renderer.material()->releaseTextureIDs(_renderer);

        for(int i = 0; i < renderer.instance_vbos.size(); ++i)
        {
            vao->popVertexBuffer();
        }
    }
}

void ComponentRenderer<MeshLightComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                            Entity entity,
                                                            const atcg::ref_ptr<Camera>& camera,
                                                            atcg::Dictionary& auxiliary) const
{
    uint32_t entity_id           = entity.entity_handle();
    TransformComponent transform = entity.getComponent<TransformComponent>();
    GeometryComponent geometry   = entity.getComponent<GeometryComponent>();

    // Actual rendering of component
    MeshLightComponent renderer = entity.getComponent<MeshLightComponent>();

    auto scene = entity.scene();

    atcg::ref_ptr<atcg::Shader> shader =
        auxiliary.getValueOr<atcg::ref_ptr<Shader>>("override_shader",
                                                    _renderer->getShaderManager()->getShader("emissive"));

    // if(renderer.visible)
    {
        auto emissive_id = _renderer->popTextureID();
        _renderer->bindTexture(emissive_id, renderer.getEmissiveTexture());
        shader->setInt("texture_emissive", emissive_id);
        shader->setFloat("emissive_scaling", renderer.intensity);
        shader->setInt("entityID", entity.entity_handle());
        GraphicsPipeline pipeline = GraphicsPipeline().setShader(shader);
        _renderer->drawVAO(geometry.graph()->getVerticesArray(),
                           camera,
                           transform.getModel(),
                           pipeline,
                           geometry.graph()->n_vertices());

        _renderer->pushTextureID(emissive_id);
    }
}

void ComponentRenderer<CameraComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                         Entity entity,
                                                         const atcg::ref_ptr<Camera>& camera,
                                                         atcg::Dictionary& auxiliary) const
{
    bool draw_cameras = auxiliary.getValueOr<bool>("draw_cameras", true);
    if(!draw_cameras)
    {
        return;
    }

    auto camera_frustum       = AssetManager::getCameraFrustumMesh();
    auto quad                 = AssetManager::getQuadMesh();
    auto shader               = _renderer->getShaderManager()->getShader("edge");
    GraphicsPipeline pipeline = GraphicsPipeline()
                                    .setShader(shader)
                                    .setPrimitiveTopology(PrimitiveTopology::ATCG_POINTS)
                                    .setRasterizerState(RasterizerState().enableCulling(false).setLineSize(2.0f));

    uint32_t entity_id = entity.entity_handle();
    shader->setInt("entityID", entity_id);
    atcg::CameraComponent& comp = entity.getComponent<CameraComponent>();
    shader->setVec3("flat_color", comp.color);
    atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(comp.camera);
    float aspect_ratio                   = cam->getAspectRatio();
    glm::mat4 scale = glm::scale(glm::vec3(aspect_ratio, 1.0f, -0.5f / glm::tan(glm::radians(cam->getFOV()) / 2.0f)) *
                                 comp.render_scale);
    glm::mat4 model = glm::inverse(cam->getView()) * scale;

    auto points = camera_frustum->getVerticesBuffer();
    _renderer->bindStorageBuffer(0, points);

    _renderer->drawVAO(camera_frustum->getEdgesArray(), camera, model, pipeline, camera_frustum->n_edges());


    if(comp.image())
    {
        uint32_t id = _renderer->popTextureID();

        auto shader = _renderer->getShaderManager()->getShader("image_display");

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                        .setRasterizerState(RasterizerState().enableCulling(false));

        model = model * glm::translate(glm::vec3(0, 0, 1)) * glm::scale(glm::vec3(0.5));
        shader->setInt("screen_texture", id);
        shader->setInt("entityID", entity_id);
        _renderer->bindTexture(id, comp.image());
        _renderer->drawVAO(quad->getVerticesArray(), camera, model, pipeline, quad->n_vertices());
        _renderer->pushTextureID(id);
    }
    else if(comp.render_preview && comp.preview)
    {
        auto shader = _renderer->getShaderManager()->getShader("image_display");

        GraphicsPipeline pipeline = GraphicsPipeline()
                                        .setShader(shader)
                                        .setPrimitiveTopology(PrimitiveTopology::ATCG_TRIANGLES)
                                        .setRasterizerState(RasterizerState().enableCulling(false));

        uint32_t id = _renderer->popTextureID();
        model       = model * glm::translate(glm::vec3(0, 0, 1)) * glm::scale(glm::vec3(0.5));
        shader->setInt("screen_texture", id);
        shader->setInt("entityID", entity_id);

        _renderer->bindTexture(id, comp.preview->getColorAttachement(0));
        _renderer->drawVAO(quad->getVerticesArray(), camera, model, pipeline, quad->n_vertices());
        _renderer->pushTextureID(id);
    }
}

void ComponentRenderer<PointLightComponent>::renderComponent(atcg::RendererSystem* _renderer,
                                                             Entity entity,
                                                             const atcg::ref_ptr<Camera>& camera,
                                                             atcg::Dictionary& auxiliary) const
{
    auto& transform   = entity.getComponent<atcg::TransformComponent>();
    auto& point_light = entity.getComponent<atcg::PointLightComponent>();

    const auto& shader = _renderer->getShaderManager()->getShader("circle");
    shader->setInt("entityID", entity.entity_handle());
    _renderer->drawCircle(transform.getPosition(), 0.1f, 1.0f, point_light.color, camera);
}
}    // namespace atcg