#include <Scene/ComponentSerializer.h>

#include <Scene/Entity.h>

#include <DataStructure/TorchUtils.h>

#include <Core/Path.h>

namespace atcg
{

namespace Serialization
{
#define ID_KEY                     "ID"
#define NAME_KEY                   "Name"
#define TRANSFORM_KEY              "Transform"
#define POSITION_KEY               "Position"
#define SCALE_KEY                  "Scale"
#define EULER_ANGLES_KEY           "EulerAngles"
#define SHADER_KEY                 "Shader"
#define MATERIAL_KEY               "Material"
#define DIFFUSE_KEY                "Diffuse"
#define DIFFUSE_TEXTURE_KEY        "DiffuseTexture"
#define NORMAL_TEXTURE_KEY         "NormalTexture"
#define ROUGHNESS_KEY              "Roughness"
#define ROUGHNESS_TEXTURE_KEY      "RoughnessTexture"
#define METALLIC_KEY               "Metallic"
#define METALLIC_TEXTURE_KEY       "MetallicTexture"
#define VERTEX_KEY                 "Vertex"
#define FRAGMENT_KEY               "Fragment"
#define GEOMETRY_KEY               "Geometry"
#define TYPE_KEY                   "Type"
#define VERTICES_KEY               "Vertices"
#define FACES_KEY                  "Faces"
#define EDGES_KEY                  "Edges"
#define GEOMETRY_KEY               "Geometry"
#define MESH_RENDERER_KEY          "MeshRenderer"
#define POINT_RENDERER_KEY         "PointRenderer"
#define POINT_SPHERE_RENDERER_KEY  "PointSphereRenderer"
#define EDGE_RENDERER_KEY          "EdgeRenderer"
#define EDGE_CYLINDER_RENDERER_KEY "EdgeCylinderRenderer"
#define INSTANCE_RENDERER_KEY      "InstanceRenderer"
#define INSTANCES_KEY              "Instances"
#define COLOR_KEY                  "Color"
#define POINT_SIZE_KEY             "PointSize"
#define RADIUS_KEY                 "Radius"
#define PERSPECTIVE_CAMERA_KEY     "PerspectiveCamera"
#define CAMERA_IMAGE_KEY           "Image"
#define ASPECT_RATIO_KEY           "AspectRatio"
#define FOVY_KEY                   "FoVy"
#define LOOKAT_KEY                 "LookAt"
#define NEAR_KEY                   "Near"
#define FAR_KEY                    "Far"
#define WIDTH_KEY                  "width"
#define HEIGHT_KEY                 "height"
#define PREVIEW_KEY                "preview"
#define OPTICAL_CENTER_KEY         "OpticalCenter"
#define POINT_LIGHT_KEY            "PointLight"
#define INTENSITY_KEY              "Intensity"
#define CAST_SHADOWS_KEY           "CastShadow"
#define RECEIVE_SHADOWS_KEY        "ReceiveShadow"
#define SCRIPT_KEY                 "Script"
#define RENDER_SCALE_KEY           "Scale"
#define LAYOUT_KEY                 "Layout"
#define PATH_KEY                   "Path"

void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size)
{
    std::ofstream summary_file(file_name, std::ios::out | std::ios::binary);
    summary_file.write(data, byte_size);
    summary_file.close();
}

std::vector<uint8_t> deserializeBuffer(const std::string& file_name)
{
    std::ifstream summary_file(file_name, std::ios::in | std::ios::binary);
    std::vector<uint8_t> buffer_char(std::istreambuf_iterator<char>(summary_file), {});
    summary_file.close();

    return buffer_char;
}

void serializeTexture(const atcg::ref_ptr<Texture2D>& texture, std::string& path, float gamma)
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

void serializeMaterial(nlohmann::json& j,
                       Entity entity,
                       const atcg::ref_ptr<Material>& material,
                       const std::string& file_path)
{
    auto diffuse_texture   = material->getDiffuseTexture();
    auto normal_texture    = material->getNormalTexture();
    auto metallic_texture  = material->getMetallicTexture();
    auto roughness_texture = material->getRoughnessTexture();

    bool use_diffuse_texture   = !(diffuse_texture->width() == 1 && diffuse_texture->height() == 1);
    bool use_normal_texture    = !(normal_texture->width() == 1 && normal_texture->height() == 1);
    bool use_metallic_texture  = !(metallic_texture->width() == 1 && metallic_texture->height() == 1);
    bool use_roughness_texture = !(roughness_texture->width() == 1 && roughness_texture->height() == 1);

    auto entity_id = entity.getComponent<IDComponent>().ID();

    auto& material_node = j[MATERIAL_KEY];

    if(use_diffuse_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_diffuse";

        serializeTexture(diffuse_texture, img_path, 1.0f / 2.2f);

        material_node[DIFFUSE_TEXTURE_KEY] = img_path;
    }
    else
    {
        auto data         = diffuse_texture->getData(atcg::CPU);
        glm::u8vec3 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>()};

        glm::vec3 c(color);
        c = c / 255.0f;

        material_node[DIFFUSE_KEY] = nlohmann::json::array({c.x, c.y, c.z});
    }

    if(use_normal_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_normal";

        serializeTexture(normal_texture, img_path);

        material_node[NORMAL_TEXTURE_KEY] = img_path;
    }

    if(use_metallic_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_metallic";

        serializeTexture(metallic_texture, img_path);

        material_node[METALLIC_TEXTURE_KEY] = img_path;
    }
    else
    {
        auto data   = metallic_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_node[METALLIC_KEY] = color;
    }

    if(use_roughness_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_roughness";

        serializeTexture(roughness_texture, img_path);

        material_node[ROUGHNESS_TEXTURE_KEY] = img_path;
    }
    else
    {
        auto data   = roughness_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_node[ROUGHNESS_KEY] = color;
    }
}

atcg::ref_ptr<Material> deserialize_material(const nlohmann::json& material_node)
{
    atcg::ref_ptr<Material> material = atcg::make_ref<Material>();

    // Diffuse
    if(material_node.contains(DIFFUSE_KEY))
    {
        std::vector<float> diffuse_color = material_node[DIFFUSE_KEY];
        material->setDiffuseColor(glm::vec4(glm::make_vec3(diffuse_color.data()), 1.0f));
    }
    else if(material_node.contains(DIFFUSE_TEXTURE_KEY))
    {
        std::string diffuse_path = material_node[DIFFUSE_TEXTURE_KEY];
        auto img                 = IO::imread(diffuse_path, 2.2f);
        auto diffuse_texture     = atcg::Texture2D::create(img);
        material->setDiffuseTexture(diffuse_texture);
    }

    // Normals
    if(material_node.contains(NORMAL_TEXTURE_KEY))
    {
        std::string normal_path = material_node[NORMAL_TEXTURE_KEY];
        auto img                = IO::imread(normal_path);
        auto normal_texture     = atcg::Texture2D::create(img);
        material->setNormalTexture(normal_texture);
    }

    // Roughness
    if(material_node.contains(ROUGHNESS_KEY))
    {
        float roughness = material_node[ROUGHNESS_KEY];
        material->setRoughness(roughness);
    }
    else if(material_node.contains(ROUGHNESS_TEXTURE_KEY))
    {
        std::string roughness_path = material_node[ROUGHNESS_TEXTURE_KEY];
        auto img                   = IO::imread(roughness_path);
        auto roughness_texture     = atcg::Texture2D::create(img);
        material->setRoughnessTexture(roughness_texture);
    }

    // Metallic
    if(material_node.contains(METALLIC_KEY))
    {
        float metallic = material_node[METALLIC_KEY];
        material->setMetallic(metallic);
    }
    else if(material_node.contains(METALLIC_TEXTURE_KEY))
    {
        std::string metallic_path = material_node[METALLIC_TEXTURE_KEY];
        auto img                  = IO::imread(metallic_path);
        auto metallic_texture     = atcg::Texture2D::create(img);
        material->setMetallicTexture(metallic_texture);
    }

    return material;
}

nlohmann::json serializeLayout(const atcg::BufferLayout& layout)
{
    nlohmann::json::array_t json_layout;
    for(auto element: layout)
    {
        nlohmann::json::array_t json_element;
        json_element.push_back((int)element.type);
        json_element.push_back(element.name);

        json_layout.push_back(json_element);
    }

    return json_layout;
}

atcg::BufferLayout deserializeLayout(nlohmann::json& layout_node)
{
    std::vector<atcg::BufferElement> elements;
    for(nlohmann::json::array_t element: layout_node)
    {
        atcg::BufferElement buffer_element((atcg::ShaderDataType)element[0], element[1]);
        elements.push_back(buffer_element);
    }

    return atcg::BufferLayout(elements);
}


void ComponentSerializer<IDComponent>::serialize_component(const std::string& file_path,
                                                           const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           IDComponent& component,
                                                           nlohmann::json& j) const
{
    j[ID_KEY] = (uint64_t)entity.getComponent<IDComponent>().ID();
}


void ComponentSerializer<NameComponent>::serialize_component(const std::string& file_path,
                                                             const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             NameComponent& component,
                                                             nlohmann::json& j) const
{
    j[NAME_KEY] = entity.getComponent<NameComponent>().name();
}


void ComponentSerializer<TransformComponent>::serialize_component(const std::string& file_path,
                                                                  const atcg::ref_ptr<Scene>& scene,
                                                                  Entity entity,
                                                                  TransformComponent& component,
                                                                  nlohmann::json& j) const
{
    glm::vec3 position                 = component.getPosition();
    glm::vec3 scale                    = component.getScale();
    glm::vec3 rotation                 = component.getRotation();
    j[TRANSFORM_KEY][POSITION_KEY]     = nlohmann::json::array({position.x, position.y, position.z});
    j[TRANSFORM_KEY][SCALE_KEY]        = nlohmann::json::array({scale.x, scale.y, scale.z});
    j[TRANSFORM_KEY][EULER_ANGLES_KEY] = nlohmann::json::array({rotation.x, rotation.y, rotation.z});
}


void ComponentSerializer<CameraComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               CameraComponent& component,
                                                               nlohmann::json& j) const
{
    atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(component.camera);

    glm::vec3 position = cam->getPosition();
    glm::vec3 look_at  = cam->getLookAt();
    glm::vec2 offset   = cam->getIntrinsics().opticalCenter();
    float n            = cam->getNear();
    float f            = cam->getFar();

    j[PERSPECTIVE_CAMERA_KEY][ASPECT_RATIO_KEY]   = cam->getAspectRatio();
    j[PERSPECTIVE_CAMERA_KEY][FOVY_KEY]           = cam->getFOV();
    j[PERSPECTIVE_CAMERA_KEY][POSITION_KEY]       = nlohmann::json::array({position.x, position.y, position.z});
    j[PERSPECTIVE_CAMERA_KEY][LOOKAT_KEY]         = nlohmann::json::array({look_at.x, look_at.y, look_at.z});
    j[PERSPECTIVE_CAMERA_KEY][NEAR_KEY]           = n;
    j[PERSPECTIVE_CAMERA_KEY][FAR_KEY]            = f;
    j[PERSPECTIVE_CAMERA_KEY][WIDTH_KEY]          = component.width;
    j[PERSPECTIVE_CAMERA_KEY][HEIGHT_KEY]         = component.height;
    j[PERSPECTIVE_CAMERA_KEY][PREVIEW_KEY]        = component.render_preview;
    j[PERSPECTIVE_CAMERA_KEY][OPTICAL_CENTER_KEY] = nlohmann::json::array({offset.x, offset.y});
    j[PERSPECTIVE_CAMERA_KEY][RENDER_SCALE_KEY]   = component.render_scale;

    if(component.image)
    {
        auto entity_id       = entity.getComponent<IDComponent>().ID();
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_cam_image";

        serializeTexture(component.image, img_path, 1.0f);

        j[PERSPECTIVE_CAMERA_KEY][CAMERA_IMAGE_KEY] = img_path;
    }
}


void ComponentSerializer<GeometryComponent>::serialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 GeometryComponent& component,
                                                                 nlohmann::json& j) const
{
    atcg::ref_ptr<Graph> graph = component.graph();

    j[GEOMETRY_KEY][TYPE_KEY] = (int)graph->type();

    IDComponent& id = entity.getComponent<IDComponent>();

    if(graph->n_vertices() != 0)
    {
        const char* buffer      = graph->getVerticesBuffer()->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID()) + ".vertices";
        serializeBuffer(buffer_name, buffer, graph->getVerticesBuffer()->size());
        j[GEOMETRY_KEY][VERTICES_KEY] = buffer_name;
        graph->getVerticesBuffer()->unmapHostPointers();
    }

    if(graph->n_faces() != 0)
    {
        const char* buffer      = graph->getFaceIndexBuffer()->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID()) + ".faces";
        serializeBuffer(buffer_name, buffer, graph->getFaceIndexBuffer()->size());
        j[GEOMETRY_KEY][FACES_KEY] = buffer_name;
        graph->getFaceIndexBuffer()->unmapHostPointers();
    }

    if(graph->n_edges() != 0)
    {
        const char* buffer      = graph->getEdgesBuffer()->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID()) + ".edges";
        serializeBuffer(buffer_name, buffer, graph->getEdgesBuffer()->size());
        j[GEOMETRY_KEY][EDGES_KEY] = buffer_name;
        graph->getEdgesBuffer()->unmapHostPointers();
    }
}


void ComponentSerializer<MeshRenderComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   MeshRenderComponent& component,
                                                                   nlohmann::json& j) const
{
    j[MESH_RENDERER_KEY][SHADER_KEY][VERTEX_KEY]   = component.shader->getVertexPath();
    j[MESH_RENDERER_KEY][SHADER_KEY][FRAGMENT_KEY] = component.shader->getFragmentPath();
    j[MESH_RENDERER_KEY][SHADER_KEY][GEOMETRY_KEY] = component.shader->getGeometryPath();
    j[MESH_RENDERER_KEY][RECEIVE_SHADOWS_KEY]      = component.receive_shadow;

    serializeMaterial(j[MESH_RENDERER_KEY], entity, component.material(), file_path);
}


void ComponentSerializer<PointRenderComponent>::serialize_component(const std::string& file_path,
                                                                    const atcg::ref_ptr<Scene>& scene,
                                                                    Entity entity,
                                                                    PointRenderComponent& component,
                                                                    nlohmann::json& j) const
{
    j[POINT_RENDERER_KEY][COLOR_KEY] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
    j[POINT_RENDERER_KEY][POINT_SIZE_KEY]           = component.point_size;
    j[POINT_RENDERER_KEY][SHADER_KEY][VERTEX_KEY]   = component.shader->getVertexPath();
    j[POINT_RENDERER_KEY][SHADER_KEY][FRAGMENT_KEY] = component.shader->getFragmentPath();
    j[POINT_RENDERER_KEY][SHADER_KEY][GEOMETRY_KEY] = component.shader->getGeometryPath();
}


void ComponentSerializer<PointSphereRenderComponent>::serialize_component(const std::string& file_path,
                                                                          const atcg::ref_ptr<Scene>& scene,
                                                                          Entity entity,
                                                                          PointSphereRenderComponent& component,
                                                                          nlohmann::json& j) const
{
    j[POINT_SPHERE_RENDERER_KEY][POINT_SIZE_KEY]           = component.point_size;
    j[POINT_SPHERE_RENDERER_KEY][SHADER_KEY][VERTEX_KEY]   = component.shader->getVertexPath();
    j[POINT_SPHERE_RENDERER_KEY][SHADER_KEY][FRAGMENT_KEY] = component.shader->getFragmentPath();
    j[POINT_SPHERE_RENDERER_KEY][SHADER_KEY][GEOMETRY_KEY] = component.shader->getGeometryPath();

    serializeMaterial(j[POINT_SPHERE_RENDERER_KEY], entity, component.material(), file_path);
}


void ComponentSerializer<EdgeRenderComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   EdgeRenderComponent& component,
                                                                   nlohmann::json& j) const
{
    j[EDGE_RENDERER_KEY][COLOR_KEY] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
}


void ComponentSerializer<EdgeCylinderRenderComponent>::serialize_component(const std::string& file_path,
                                                                           const atcg::ref_ptr<Scene>& scene,
                                                                           Entity entity,
                                                                           EdgeCylinderRenderComponent& component,
                                                                           nlohmann::json& j) const
{
    j[EDGE_CYLINDER_RENDERER_KEY][RADIUS_KEY] = component.radius;

    serializeMaterial(j[EDGE_CYLINDER_RENDERER_KEY], entity, component.material(), file_path);
}


void ComponentSerializer<InstanceRenderComponent>::serialize_component(const std::string& file_path,
                                                                       const atcg::ref_ptr<Scene>& scene,
                                                                       Entity entity,
                                                                       InstanceRenderComponent& component,
                                                                       nlohmann::json& j) const
{
    auto shader = component.shader ? component.shader : atcg::ShaderManager::getShader("instanced");
    j[INSTANCE_RENDERER_KEY][SHADER_KEY][VERTEX_KEY]   = shader->getVertexPath();
    j[INSTANCE_RENDERER_KEY][SHADER_KEY][FRAGMENT_KEY] = shader->getFragmentPath();
    j[INSTANCE_RENDERER_KEY][SHADER_KEY][GEOMETRY_KEY] = shader->getGeometryPath();
    serializeMaterial(j[INSTANCE_RENDERER_KEY], entity, component.material(), file_path);

    IDComponent& id = entity.getComponent<IDComponent>();
    nlohmann::json::array_t buffers;
    for(int i = 0; i < component.instance_vbos.size(); ++i)
    {
        const char* buffer      = component.instance_vbos[i]->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID()) + ".instance_" + std::to_string(i);
        serializeBuffer(buffer_name, buffer, component.instance_vbos[i]->size());
        nlohmann::json json_buffer;
        json_buffer[PATH_KEY]   = buffer_name;
        json_buffer[LAYOUT_KEY] = serializeLayout(component.instance_vbos[i]->getLayout());
        buffers.push_back(json_buffer);
        component.instance_vbos[i]->unmapHostPointers();
    }

    j[INSTANCE_RENDERER_KEY][INSTANCES_KEY] = buffers;
}


void ComponentSerializer<PointLightComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   PointLightComponent& component,
                                                                   nlohmann::json& j) const
{
    j[POINT_LIGHT_KEY][INTENSITY_KEY] = component.intensity;
    j[POINT_LIGHT_KEY][COLOR_KEY] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
    j[POINT_LIGHT_KEY][CAST_SHADOWS_KEY] = component.cast_shadow;
}


void ComponentSerializer<ScriptComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               ScriptComponent& component,
                                                               nlohmann::json& j) const
{
    std::string path = "";
    if(component.script)
    {
        path = component.script->getFilePath().string();
    }

    j[SCRIPT_KEY] = path;
}


void ComponentSerializer<IDComponent>::deserialize_component(const std::string& file_path,
                                                             const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             nlohmann::json& j) const
{
    if(!j.contains(ID_KEY))
    {
        return;
    }

    entity.addOrReplaceComponent<IDComponent>((uint64_t)j[ID_KEY]);
}


void ComponentSerializer<NameComponent>::deserialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               nlohmann::json& j) const
{
    if(!j.contains(NAME_KEY))
    {
        return;
    }

    entity.addOrReplaceComponent<NameComponent>(j[NAME_KEY]);
}


void ComponentSerializer<TransformComponent>::deserialize_component(const std::string& file_path,
                                                                    const atcg::ref_ptr<Scene>& scene,
                                                                    Entity entity,
                                                                    nlohmann::json& j) const
{
    if(!j.contains(TRANSFORM_KEY))
    {
        return;
    }

    std::vector<float> position = j[TRANSFORM_KEY].value(POSITION_KEY, std::vector<float> {0.0f, 0.0f, 0.0f});
    std::vector<float> scale    = j[TRANSFORM_KEY].value(SCALE_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    std::vector<float> rotation = j[TRANSFORM_KEY].value(EULER_ANGLES_KEY, std::vector<float> {0.0f, 0.0f, 0.0f});

    entity.addComponent<atcg::TransformComponent>(glm::make_vec3(position.data()),
                                                  glm::make_vec3(scale.data()),
                                                  glm::make_vec3(rotation.data()));
}


void ComponentSerializer<CameraComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(!j.contains(PERSPECTIVE_CAMERA_KEY))
    {
        return;
    }

    float aspect_ratio          = j[PERSPECTIVE_CAMERA_KEY].value(ASPECT_RATIO_KEY, 1.0f);
    float fov                   = j[PERSPECTIVE_CAMERA_KEY].value(FOVY_KEY, 60.0f);
    float n                     = j[PERSPECTIVE_CAMERA_KEY].value(NEAR_KEY, 0.01f);
    float f                     = j[PERSPECTIVE_CAMERA_KEY].value(FAR_KEY, 1000.0f);
    std::vector<float> position = j[PERSPECTIVE_CAMERA_KEY].value(POSITION_KEY, std::vector<float> {0.0f, 0.0f, -1.0f});
    std::vector<float> lookat   = j[PERSPECTIVE_CAMERA_KEY].value(LOOKAT_KEY, std::vector<float> {0.0f, 0.0f, 0.0f});
    std::vector<float> offset   = j[PERSPECTIVE_CAMERA_KEY].value(OPTICAL_CENTER_KEY, std::vector<float> {0.0f, 0.0f});

    CameraExtrinsics extrinsics(glm::make_vec3(position.data()), glm::make_vec3(lookat.data()));
    CameraIntrinsics intrinsics(aspect_ratio, fov, n, f);
    intrinsics.setOpticalCenter(glm::make_vec2(offset.data()));

    auto cam = atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics);

    auto& component          = entity.addComponent<CameraComponent>(cam);
    component.width          = j[PERSPECTIVE_CAMERA_KEY].value(WIDTH_KEY, 1024);
    component.height         = j[PERSPECTIVE_CAMERA_KEY].value(HEIGHT_KEY, 1024);
    component.render_preview = j[PERSPECTIVE_CAMERA_KEY].value(PREVIEW_KEY, false);
    component.render_scale   = j[PERSPECTIVE_CAMERA_KEY].value(RENDER_SCALE_KEY, 1.0f);


    if(j[PERSPECTIVE_CAMERA_KEY].contains(CAMERA_IMAGE_KEY))
    {
        std::string diffuse_path = j[PERSPECTIVE_CAMERA_KEY][CAMERA_IMAGE_KEY];
        auto img                 = IO::imread(diffuse_path);
        component.image          = atcg::Texture2D::create(img);
    }
}


void ComponentSerializer<GeometryComponent>::deserialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   nlohmann::json& j) const
{
    if(!j.contains(GEOMETRY_KEY))
    {
        return;
    }

    auto& geometry       = entity.addComponent<GeometryComponent>();
    atcg::GraphType type = (atcg::GraphType)(int)j[GEOMETRY_KEY][TYPE_KEY];

    switch(type)
    {
        case atcg::GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH:
        {
            std::string vertex_path           = j[GEOMETRY_KEY][VERTICES_KEY];
            std::string faces_path            = j[GEOMETRY_KEY][FACES_KEY];
            std::vector<uint8_t> vertices_raw = deserializeBuffer(vertex_path);
            std::vector<uint8_t> faces_raw    = deserializeBuffer(faces_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            auto faces = atcg::createHostTensorFromPointer((int32_t*)faces_raw.data(),
                                                           {(int)(faces_raw.size() / sizeof(glm::u32vec3)), 3});

            geometry.graph() = Graph::createTriangleMesh(vertices, faces);
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_POINTCLOUD:
        {
            std::string vertex_path           = j[GEOMETRY_KEY][VERTICES_KEY];
            std::vector<uint8_t> vertices_raw = deserializeBuffer(vertex_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            geometry.graph() = Graph::createPointCloud(vertices);
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_GRAPH:
        {
            std::string vertex_path           = j[GEOMETRY_KEY][VERTICES_KEY];
            std::string edges_path            = j[GEOMETRY_KEY][EDGES_KEY];
            std::vector<uint8_t> vertices_raw = deserializeBuffer(vertex_path);
            std::vector<uint8_t> edges_raw    = deserializeBuffer(edges_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            auto edges = atcg::createHostTensorFromPointer(
                (float*)edges_raw.data(),
                {(int)(edges_raw.size() / sizeof(Edge)), atcg::EdgeSpecification::EDGE_SIZE});

            geometry.graph() = Graph::createGraph(vertices, edges);
        }
        break;
        default:
        {
            ATCG_ERROR("Unknown graph type: {0}", (int)type);
        }
        break;
    }
}


void ComponentSerializer<MeshRenderComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(MESH_RENDERER_KEY))
    {
        return;
    }

    auto& renderer          = j[MESH_RENDERER_KEY];
    auto& renderComponent   = entity.addComponent<MeshRenderComponent>();
    std::string vertex_path = renderer[SHADER_KEY].value(VERTEX_KEY, (atcg::shader_directory() / "base.vs").string());
    std::string fragment_path =
        renderer[SHADER_KEY].value(FRAGMENT_KEY, (atcg::shader_directory() / "base.fs").string());
    std::string geometry_path = renderer[SHADER_KEY].value(GEOMETRY_KEY, "");

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

    if(renderer.contains(MATERIAL_KEY))
    {
        auto material_node               = renderer[MATERIAL_KEY];
        atcg::ref_ptr<Material> material = deserialize_material(material_node);
        renderComponent.setMaterial(material);
    }

    renderComponent.receive_shadow = renderer.value(RECEIVE_SHADOWS_KEY, true);
}


void ComponentSerializer<PointRenderComponent>::deserialize_component(const std::string& file_path,
                                                                      const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      nlohmann::json& j) const
{
    if(!j.contains(POINT_RENDERER_KEY))
    {
        return;
    }

    auto& renderer             = j[POINT_RENDERER_KEY];
    auto& renderComponent      = entity.addComponent<PointRenderComponent>();
    std::vector<float> color   = renderer.value(COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    renderComponent.color      = glm::make_vec3(color.data());
    renderComponent.point_size = renderer.value(POINT_SIZE_KEY, 1.0f);
    std::string vertex_path = renderer[SHADER_KEY].value(VERTEX_KEY, (atcg::shader_directory() / "base.vs").string());
    std::string fragment_path =
        renderer[SHADER_KEY].value(FRAGMENT_KEY, (atcg::shader_directory() / "base.fs").string());
    std::string geometry_path = renderer[SHADER_KEY].value(GEOMETRY_KEY, "");

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


void ComponentSerializer<PointSphereRenderComponent>::deserialize_component(const std::string& file_path,
                                                                            const atcg::ref_ptr<Scene>& scene,
                                                                            Entity entity,
                                                                            nlohmann::json& j) const
{
    if(!j.contains(POINT_SPHERE_RENDERER_KEY))
    {
        return;
    }

    auto& renderer             = j[POINT_SPHERE_RENDERER_KEY];
    auto& renderComponent      = entity.addComponent<PointSphereRenderComponent>();
    renderComponent.point_size = renderer.value(POINT_SIZE_KEY, 0.1f);
    std::string vertex_path = renderer[SHADER_KEY].value(VERTEX_KEY, (atcg::shader_directory() / "base.vs").string());
    std::string fragment_path =
        renderer[SHADER_KEY].value(FRAGMENT_KEY, (atcg::shader_directory() / "base.fs").string());
    std::string geometry_path = renderer[SHADER_KEY].value(GEOMETRY_KEY, "");

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


    if(renderer.contains(MATERIAL_KEY))
    {
        auto material_node               = renderer[MATERIAL_KEY];
        atcg::ref_ptr<Material> material = deserialize_material(material_node);
        renderComponent.setMaterial(material);
    }
}


void ComponentSerializer<EdgeRenderComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(EDGE_RENDERER_KEY))
    {
        return;
    }

    auto& renderer           = j[EDGE_RENDERER_KEY];
    auto& renderComponent    = entity.addComponent<EdgeRenderComponent>();
    std::vector<float> color = renderer.value(COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    renderComponent.color    = glm::make_vec3(color.data());
}


void ComponentSerializer<EdgeCylinderRenderComponent>::deserialize_component(const std::string& file_path,
                                                                             const atcg::ref_ptr<Scene>& scene,
                                                                             Entity entity,
                                                                             nlohmann::json& j) const
{
    if(!j.contains(EDGE_CYLINDER_RENDERER_KEY))
    {
        return;
    }

    auto& renderer         = j[EDGE_CYLINDER_RENDERER_KEY];
    auto& renderComponent  = entity.addComponent<EdgeCylinderRenderComponent>();
    renderComponent.radius = renderer.value(RADIUS_KEY, 0.001f);


    if(renderer.contains(MATERIAL_KEY))
    {
        auto& material_node              = renderer[MATERIAL_KEY];
        atcg::ref_ptr<Material> material = deserialize_material(material_node);
        renderComponent.setMaterial(material);
    }
}


void ComponentSerializer<InstanceRenderComponent>::deserialize_component(const std::string& file_path,
                                                                         const atcg::ref_ptr<Scene>& scene,
                                                                         Entity entity,
                                                                         nlohmann::json& j) const
{
    if(!j.contains(INSTANCE_RENDERER_KEY))
    {
        return;
    }

    auto& renderer        = j[INSTANCE_RENDERER_KEY];
    auto& renderComponent = entity.addComponent<InstanceRenderComponent>();
    if(renderer.contains(MATERIAL_KEY))
    {
        auto& material_node              = renderer[MATERIAL_KEY];
        atcg::ref_ptr<Material> material = deserialize_material(material_node);
        renderComponent.setMaterial(material);
    }

    std::string vertex_path =
        renderer[SHADER_KEY].value(VERTEX_KEY, (atcg::shader_directory() / "instanced.vs").string());
    std::string fragment_path =
        renderer[SHADER_KEY].value(FRAGMENT_KEY, (atcg::shader_directory() / "instanced.fs").string());
    std::string geometry_path = renderer[SHADER_KEY].value(GEOMETRY_KEY, "");

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

    if(renderer.contains(INSTANCES_KEY))
    {
        nlohmann::json::array_t instances = renderer[INSTANCES_KEY];
        renderComponent.instance_vbos.reserve(instances.size());

        for(auto instance: instances)
        {
            std::string path                      = instance[PATH_KEY];
            atcg::BufferLayout layout             = deserializeLayout(instance[LAYOUT_KEY]);
            std::vector<uint8_t> buffer           = deserializeBuffer(path);
            atcg::ref_ptr<atcg::VertexBuffer> vbo = atcg::make_ref<atcg::VertexBuffer>(buffer.data(), buffer.size());
            vbo->setLayout(layout);
            renderComponent.addInstanceBuffer(vbo);
        }
    }
}


void ComponentSerializer<PointLightComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(POINT_LIGHT_KEY))
    {
        return;
    }

    auto& point_light           = j[POINT_LIGHT_KEY];
    auto& renderComponent       = entity.addComponent<PointLightComponent>();
    renderComponent.intensity   = point_light.value(INTENSITY_KEY, 1.0f);
    auto color                  = point_light.value(COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    renderComponent.color       = glm::make_vec3(color.data());
    renderComponent.cast_shadow = point_light.value(CAST_SHADOWS_KEY, true);
}


void ComponentSerializer<ScriptComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(!j.contains(SCRIPT_KEY))
    {
        return;
    }

    auto& script = entity.addComponent<ScriptComponent>();

    std::string path = j[SCRIPT_KEY];
    if(path != "")
    {
        script.script = atcg::make_ref<PythonScript>(path);
        script.script->init(scene, entity);
        script.script->onAttach();
    }
}
}    // namespace Serialization
}    // namespace atcg