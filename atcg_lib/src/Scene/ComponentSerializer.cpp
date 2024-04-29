#include <Scene/ComponentSerializer.h>

#include <Scene/Entity.h>

#include <DataStructure/TorchUtils.h>

namespace atcg
{
void ComponentSerializer::serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size)
{
    std::ofstream summary_file(file_name, std::ios::out | std::ios::binary);
    summary_file.write(data, byte_size);
    summary_file.close();
}

std::vector<uint8_t> ComponentSerializer::deserializeBuffer(const std::string& file_name)
{
    std::ifstream summary_file(file_name, std::ios::in | std::ios::binary);
    std::vector<uint8_t> buffer_char(std::istreambuf_iterator<char>(summary_file), {});
    summary_file.close();

    return buffer_char;
}

void ComponentSerializer::serializeTexture(const atcg::ref_ptr<Texture2D>& texture, std::string& path, float gamma)
{
    torch::Tensor texture_data = texture->getData(atcg::CPU);

    Image img(texture_data);

    std::string file_ending = ".png";
    if(img.isHDR())
    {
        file_ending = ".hdr";
    }

    path = path + file_ending;
    img.applyGamma(gamma);
    img.store(path);
}

void ComponentSerializer::serializeMaterial(nlohmann::json& j,
                                            Entity entity,
                                            const Material& material,
                                            const std::string& file_path)
{
    auto diffuse_texture   = material.getDiffuseTexture();
    auto normal_texture    = material.getNormalTexture();
    auto metallic_texture  = material.getMetallicTexture();
    auto roughness_texture = material.getRoughnessTexture();

    bool use_diffuse_texture   = !(diffuse_texture->width() == 1 && diffuse_texture->height() == 1);
    bool use_normal_texture    = !(normal_texture->width() == 1 && normal_texture->height() == 1);
    bool use_metallic_texture  = !(metallic_texture->width() == 1 && metallic_texture->height() == 1);
    bool use_roughness_texture = !(roughness_texture->width() == 1 && roughness_texture->height() == 1);

    auto entity_id = entity.getComponent<IDComponent>().ID;

    if(use_diffuse_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_diffuse";

        serializeTexture(diffuse_texture, img_path, 1.0f / 2.2f);

        j["Material"]["DiffuseTexture"] = img_path;
    }
    else
    {
        auto data         = diffuse_texture->getData(atcg::CPU);
        glm::u8vec3 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>()};

        glm::vec3 c(color);
        c = c / 255.0f;

        j["Material"]["Diffuse"] = nlohmann::json::array({c.x, c.y, c.z});
    }

    if(use_normal_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_normal";

        serializeTexture(normal_texture, img_path);

        j["Material"]["NormalTexture"] = img_path;
    }

    if(use_metallic_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_metallic";

        serializeTexture(metallic_texture, img_path);

        j["Material"]["MetallicTexture"] = img_path;
    }
    else
    {
        auto data   = metallic_texture->getData(atcg::CPU);
        float color = data.item<float>();

        j["Material"]["Metallic"] = color;
    }

    if(use_roughness_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_roughness";

        serializeTexture(roughness_texture, img_path);

        j["Material"]["RoughnessTexture"] = img_path;
    }
    else
    {
        auto data   = roughness_texture->getData(atcg::CPU);
        float color = data.item<float>();

        j["Material"]["Roughness"] = color;
    }
}

Material ComponentSerializer::deserialize_material(const nlohmann::json& material_node)
{
    Material material;

    // Diffuse
    if(material_node.contains("Diffuse"))
    {
        std::vector<float> diffuse_color = material_node["Diffuse"];
        material.setDiffuseColor(glm::vec4(glm::make_vec3(diffuse_color.data()), 1.0f));
    }
    else if(material_node.contains("DiffuseTexture"))
    {
        std::string diffuse_path = material_node["DiffuseTexture"];
        ATCG_TRACE(diffuse_path);
        auto img             = IO::imread(diffuse_path, 2.2f);
        auto diffuse_texture = atcg::Texture2D::create(img);
        material.setDiffuseTexture(diffuse_texture);
    }

    // Normals
    if(material_node.contains("NormalTexture"))
    {
        std::string normal_path = material_node["NormalTexture"];
        auto img                = IO::imread(normal_path);
        auto normal_texture     = atcg::Texture2D::create(img);
        material.setNormalTexture(normal_texture);
    }

    // Roughness
    if(material_node.contains("Roughness"))
    {
        float roughness = material_node["Roughness"];
        material.setRoughness(roughness);
    }
    else if(material_node.contains("RoughnessTexture"))
    {
        std::string roughness_path = material_node["RoughnessTexture"];
        auto img                   = IO::imread(roughness_path);
        auto roughness_texture     = atcg::Texture2D::create(img);
        material.setRoughnessTexture(roughness_texture);
    }

    // Metallic
    if(material_node.contains("Metallic"))
    {
        float metallic = material_node["Metallic"];
        material.setMetallic(metallic);
    }
    else if(material_node.contains("MetallicTexture"))
    {
        std::string metallic_path = material_node["MetallicTexture"];
        auto img                  = IO::imread(metallic_path);
        auto metallic_texture     = atcg::Texture2D::create(img);
        material.setMetallicTexture(metallic_texture);
    }

    return material;
}

template<typename T>
void ComponentSerializer::serialize_component(const std::string& file_path,
                                              Entity entity,
                                              T& component,
                                              nlohmann::json& j)
{
}

template<>
void ComponentSerializer::serialize_component<IDComponent>(const std::string& file_path,
                                                           Entity entity,
                                                           IDComponent& component,
                                                           nlohmann::json& j)
{
    j["ID"] = (uint64_t)entity.getComponent<IDComponent>().ID;
}

template<>
void ComponentSerializer::serialize_component<NameComponent>(const std::string& file_path,
                                                             Entity entity,
                                                             NameComponent& component,
                                                             nlohmann::json& j)
{
    j["Name"] = entity.getComponent<NameComponent>().name;
}

template<>
void ComponentSerializer::serialize_component<TransformComponent>(const std::string& file_path,
                                                                  Entity entity,
                                                                  TransformComponent& component,
                                                                  nlohmann::json& j)
{
    glm::vec3 position            = component.getPosition();
    glm::vec3 scale               = component.getScale();
    glm::vec3 rotation            = component.getRotation();
    j["Transform"]["Position"]    = nlohmann::json::array({position.x, position.y, position.z});
    j["Transform"]["Scale"]       = nlohmann::json::array({scale.x, scale.y, scale.z});
    j["Transform"]["EulerAngles"] = nlohmann::json::array({rotation.x, rotation.y, rotation.z});
}

template<>
void ComponentSerializer::serialize_component<CameraComponent>(const std::string& file_path,
                                                               Entity entity,
                                                               CameraComponent& component,
                                                               nlohmann::json& j)
{
    atcg::ref_ptr<PerspectiveCamera> cam = component.camera;

    j["PerspectiveCamera"]["AspectRatio"] = cam->getAspectRatio();
    j["PerspectiveCamera"]["FoVy"]        = cam->getFOV();
}

template<>
void ComponentSerializer::serialize_component<GeometryComponent>(const std::string& file_path,
                                                                 Entity entity,
                                                                 GeometryComponent& component,
                                                                 nlohmann::json& j)
{
    atcg::ref_ptr<Graph> graph = component.graph;

    j["Geometry"]["Type"] = (int)graph->type();

    IDComponent& id = entity.getComponent<IDComponent>();

    if(graph->n_vertices() != 0)
    {
        const char* buffer      = graph->getVerticesBuffer()->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".vertices";
        serializeBuffer(buffer_name, buffer, graph->getVerticesBuffer()->size());
        j["Geometry"]["Vertices"] = buffer_name;
        graph->getVerticesBuffer()->unmapHostPointers();
    }

    if(graph->n_faces() != 0)
    {
        const char* buffer      = graph->getFaceIndexBuffer()->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".faces";
        serializeBuffer(buffer_name, buffer, graph->getFaceIndexBuffer()->size());
        j["Geometry"]["Faces"] = buffer_name;
        graph->getFaceIndexBuffer()->unmapHostPointers();
    }

    if(graph->n_edges() != 0)
    {
        const char* buffer      = graph->getEdgesBuffer()->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".edges";
        serializeBuffer(buffer_name, buffer, graph->getEdgesBuffer()->size());
        j["Geometry"]["Edges"] = buffer_name;
        graph->getEdgesBuffer()->unmapHostPointers();
    }
}

template<>
void ComponentSerializer::serialize_component<MeshRenderComponent>(const std::string& file_path,
                                                                   Entity entity,
                                                                   MeshRenderComponent& component,
                                                                   nlohmann::json& j)
{
    j["MeshRenderer"]["Shader"]["Vertex"]   = component.shader->getVertexPath();
    j["MeshRenderer"]["Shader"]["Fragment"] = component.shader->getFragmentPath();
    j["MeshRenderer"]["Shader"]["Geometry"] = component.shader->getGeometryPath();

    serializeMaterial(j["MeshRenderer"], entity, component.material, file_path);
}

template<>
void ComponentSerializer::serialize_component<PointRenderComponent>(const std::string& file_path,
                                                                    Entity entity,
                                                                    PointRenderComponent& component,
                                                                    nlohmann::json& j)
{
    j["PointRenderer"]["Color"]     = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
    j["PointRenderer"]["PointSize"] = component.point_size;
    j["PointRenderer"]["Shader"]["Vertex"]   = component.shader->getVertexPath();
    j["PointRenderer"]["Shader"]["Fragment"] = component.shader->getFragmentPath();
    j["PointRenderer"]["Shader"]["Geometry"] = component.shader->getGeometryPath();
}

template<>
void ComponentSerializer::serialize_component<PointSphereRenderComponent>(const std::string& file_path,
                                                                          Entity entity,
                                                                          PointSphereRenderComponent& component,
                                                                          nlohmann::json& j)
{
    j["PointSphereRenderer"]["PointSize"]          = component.point_size;
    j["PointSphereRenderer"]["Shader"]["Vertex"]   = component.shader->getVertexPath();
    j["PointSphereRenderer"]["Shader"]["Fragment"] = component.shader->getFragmentPath();
    j["PointSphereRenderer"]["Shader"]["Geometry"] = component.shader->getGeometryPath();

    serializeMaterial(j["PointSphereRenderer"], entity, component.material, file_path);
}

template<>
void ComponentSerializer::serialize_component<EdgeRenderComponent>(const std::string& file_path,
                                                                   Entity entity,
                                                                   EdgeRenderComponent& component,
                                                                   nlohmann::json& j)
{
    j["EdgeRenderer"]["Color"] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
}

template<>
void ComponentSerializer::serialize_component<EdgeCylinderRenderComponent>(const std::string& file_path,
                                                                           Entity entity,
                                                                           EdgeCylinderRenderComponent& component,
                                                                           nlohmann::json& j)
{
    j["EdgeCylinderRenderer"]["Radius"] = component.radius;

    serializeMaterial(j["EdgeCylinderRenderer"], entity, component.material, file_path);
}

template<typename T>
void ComponentSerializer::deserialize_component(const std::string& file_path, Entity entity, nlohmann::json& j)
{
}

template<>
void ComponentSerializer::deserialize_component<IDComponent>(const std::string& file_path,
                                                             Entity entity,
                                                             nlohmann::json& j)
{
    if(!j.contains("ID"))
    {
        return;
    }

    auto& component = entity.getComponent<IDComponent>();
    component.ID    = (uint64_t)j["ID"];
}

template<>
void ComponentSerializer::deserialize_component<NameComponent>(const std::string& file_path,
                                                               Entity entity,
                                                               nlohmann::json& j)
{
    if(!j.contains("Name"))
    {
        return;
    }

    auto& component = entity.getComponent<NameComponent>();
    component.name  = j["Name"];
}

template<>
void ComponentSerializer::deserialize_component<TransformComponent>(const std::string& file_path,
                                                                    Entity entity,
                                                                    nlohmann::json& j)
{
    if(!j.contains("Transform"))
    {
        return;
    }

    std::vector<float> position = j["Transform"]["Position"];
    std::vector<float> scale    = j["Transform"]["Scale"];
    std::vector<float> rotation = j["Transform"]["EulerAngles"];

    entity.addComponent<atcg::TransformComponent>(glm::make_vec3(position.data()),
                                                  glm::make_vec3(scale.data()),
                                                  glm::make_vec3(rotation.data()));
}

template<>
void ComponentSerializer::deserialize_component<CameraComponent>(const std::string& file_path,
                                                                 Entity entity,
                                                                 nlohmann::json& j)
{
    if(!j.contains("PerspectiveCamera"))
    {
        return;
    }

    float aspect_ratio = j["PerspectiveCamera"]["AspectRatio"];
    float fov          = j["PerspectiveCamera"]["FoVy"];
    auto& camera       = entity.addComponent<CameraComponent>(
        atcg::make_ref<atcg::PerspectiveCamera>(aspect_ratio, glm::vec3(0), glm::vec3(0, 0, 1)));
    atcg::ref_ptr<PerspectiveCamera> cam = camera.camera;
    // TODO: Keep consistent
    cam->setFOV(fov);
    // Should always have a transform, otherwise construct default camera
    if(entity.hasComponent<TransformComponent>())
    {
        cam->setFromTransform(entity.getComponent<TransformComponent>().getModel());
    }
}


template<>
void ComponentSerializer::deserialize_component<GeometryComponent>(const std::string& file_path,
                                                                   Entity entity,
                                                                   nlohmann::json& j)
{
    if(!j.contains("Geometry"))
    {
        return;
    }

    auto& geometry       = entity.addComponent<GeometryComponent>();
    atcg::GraphType type = (atcg::GraphType)(int)j["Geometry"]["Type"];

    switch(type)
    {
        case atcg::GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH:
        {
            std::string vertex_path           = j["Geometry"]["Vertices"];
            std::string faces_path            = j["Geometry"]["Faces"];
            std::vector<uint8_t> vertices_raw = deserializeBuffer(vertex_path);
            std::vector<uint8_t> faces_raw    = deserializeBuffer(faces_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            auto faces = atcg::createHostTensorFromPointer((int32_t*)faces_raw.data(),
                                                           {(int)(faces_raw.size() / sizeof(glm::u32vec3)), 3});

            geometry.graph = Graph::createTriangleMesh(vertices, faces);
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_POINTCLOUD:
        {
            std::string vertex_path           = j["Geometry"]["Vertices"];
            std::vector<uint8_t> vertices_raw = deserializeBuffer(vertex_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            geometry.graph = Graph::createPointCloud(vertices);
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_GRAPH:
        {
            std::string vertex_path           = j["Geometry"]["Vertices"];
            std::string edges_path            = j["Geometry"]["Edges"];
            std::vector<uint8_t> vertices_raw = deserializeBuffer(vertex_path);
            std::vector<uint8_t> edges_raw    = deserializeBuffer(edges_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            auto edges = atcg::createHostTensorFromPointer(
                (float*)edges_raw.data(),
                {(int)(edges_raw.size() / sizeof(Edge)), atcg::EdgeSpecification::EDGE_SIZE});

            geometry.graph = Graph::createGraph(vertices, edges);
        }
        break;
        default:
        {
            ATCG_ERROR("Unknown graph type: {0}", (int)type);
        }
        break;
    }
}

template<>
void ComponentSerializer::deserialize_component<MeshRenderComponent>(const std::string& file_path,
                                                                     Entity entity,
                                                                     nlohmann::json& j)
{
    if(!j.contains("MeshRenderer"))
    {
        return;
    }

    auto& renderer            = j["MeshRenderer"];
    auto& renderComponent     = entity.addComponent<MeshRenderComponent>();
    std::string vertex_path   = renderer["Shader"]["Vertex"];
    std::string fragment_path = renderer["Shader"]["Fragment"];
    std::string geometry_path = renderer["Shader"]["Geometry"];

    std::string shader_name = vertex_path.substr(vertex_path.find_last_of('/') + 1);
    shader_name             = shader_name.substr(0, shader_name.find_first_of('.'));

    if(ShaderManager::hasShader(shader_name))
    {
        renderComponent.shader = ShaderManager::getShader(shader_name);
    }
    else if(geometry_path != "")
    {
        renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path, geometry_path);
    }
    else
    {
        renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path);
    }

    if(renderer.contains("Material"))
    {
        auto material_node       = renderer["Material"];
        Material material        = deserialize_material(material_node);
        renderComponent.material = material;
    }
}

template<>
void ComponentSerializer::deserialize_component<PointRenderComponent>(const std::string& file_path,
                                                                      Entity entity,
                                                                      nlohmann::json& j)
{
    if(!j.contains("PointRenderer"))
    {
        return;
    }

    auto& renderer             = j["PointRenderer"];
    auto& renderComponent      = entity.addComponent<PointRenderComponent>();
    std::vector<float> color   = renderer["Color"];
    renderComponent.color      = glm::make_vec3(color.data());
    renderComponent.point_size = renderer["PointSize"];
    std::string vertex_path    = renderer["Shader"]["Vertex"];
    std::string fragment_path  = renderer["Shader"]["Fragment"];
    std::string geometry_path  = renderer["Shader"]["Geometry"];

    std::string shader_name = vertex_path.substr(vertex_path.find_last_of('/') + 1);
    shader_name             = shader_name.substr(0, shader_name.find_first_of('.'));

    if(ShaderManager::hasShader(shader_name))
    {
        renderComponent.shader = ShaderManager::getShader(shader_name);
    }
    else if(geometry_path != "")
    {
        renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path, geometry_path);
    }
    else
    {
        renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path);
    }
}

template<>
void ComponentSerializer::deserialize_component<PointSphereRenderComponent>(const std::string& file_path,
                                                                            Entity entity,
                                                                            nlohmann::json& j)
{
    if(!j.contains("PointSphereRenderer"))
    {
        return;
    }

    auto& renderer             = j["PointSphereRenderer"];
    auto& renderComponent      = entity.addComponent<PointSphereRenderComponent>();
    renderComponent.point_size = renderer["PointSize"];
    std::string vertex_path    = renderer["Shader"]["Vertex"];
    std::string fragment_path  = renderer["Shader"]["Fragment"];
    std::string geometry_path  = renderer["Shader"]["Geometry"];

    std::string shader_name = vertex_path.substr(vertex_path.find_last_of('/') + 1);
    shader_name             = shader_name.substr(0, shader_name.find_first_of('.'));

    if(ShaderManager::hasShader(shader_name))
    {
        renderComponent.shader = ShaderManager::getShader(shader_name);
    }
    else if(geometry_path != "")
    {
        renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path, geometry_path);
    }
    else
    {
        renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path);
    }


    if(renderer.contains("Material"))
    {
        auto material_node       = renderer["Material"];
        Material material        = deserialize_material(material_node);
        renderComponent.material = material;
    }
}

template<>
void ComponentSerializer::deserialize_component<EdgeRenderComponent>(const std::string& file_path,
                                                                     Entity entity,
                                                                     nlohmann::json& j)
{
    if(!j.contains("EdgeRenderer"))
    {
        return;
    }

    auto& renderer           = j["EdgeRenderer"];
    auto& renderComponent    = entity.addComponent<EdgeRenderComponent>();
    std::vector<float> color = renderer["Color"];
    renderComponent.color    = glm::make_vec3(color.data());
}

template<>
void ComponentSerializer::deserialize_component<EdgeCylinderRenderComponent>(const std::string& file_path,
                                                                             Entity entity,
                                                                             nlohmann::json& j)
{
    if(!j.contains("EdgeCylinderRenderer"))
    {
        return;
    }

    auto& renderer         = j["EdgeCylinderRenderer"];
    auto& renderComponent  = entity.addComponent<EdgeCylinderRenderComponent>();
    renderComponent.radius = renderer["Radius"];


    if(renderer.contains("Material"))
    {
        auto& material_node      = renderer["Material"];
        Material material        = deserialize_material(material_node);
        renderComponent.material = material;
    }
}
}    // namespace atcg