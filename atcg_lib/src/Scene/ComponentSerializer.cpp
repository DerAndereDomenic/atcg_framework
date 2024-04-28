#include <Scene/ComponentSerializer.h>

#include <Scene/Entity.h>

namespace atcg
{
void ComponentSerializer::serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size)
{
    std::ofstream summary_file(file_name, std::ios::out | std::ios::binary);
    summary_file.write(data, byte_size);
    summary_file.close();
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

        serializeTexture(roughness_texture, img_path, 1.0f / 2.2f);

        j["Material"]["RoughnessTexture"] = img_path;
    }
    else
    {
        auto data   = roughness_texture->getData(atcg::CPU);
        float color = data.item<float>();

        j["Material"]["Roughness"] = color;
    }
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
}    // namespace atcg