#include <Scene/Serializer.h>

#include <Core/Application.h>

#include <Scene/Entity.h>
#include <Scene/Components.h>

#include <yaml-cpp/yaml.h>

namespace YAML
{
template<>
struct convert<glm::vec2>
{
    static Node encode(const glm::vec2& rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.SetStyle(EmitterStyle::Flow);
        return node;
    }

    static bool decode(const Node& node, glm::vec2& rhs)
    {
        if(!node.IsSequence() || node.size() != 2) { return false; }

        rhs.x = node[0].as<float>();
        rhs.y = node[1].as<float>();
        return true;
    }
};

template<>
struct convert<glm::vec3>
{
    static Node encode(const glm::vec3& rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        node.SetStyle(EmitterStyle::Flow);
        return node;
    }

    static bool decode(const Node& node, glm::vec3& rhs)
    {
        if(!node.IsSequence() || node.size() != 3) { return false; }

        rhs.x = node[0].as<float>();
        rhs.y = node[1].as<float>();
        rhs.z = node[2].as<float>();
        return true;
    }
};

template<>
struct convert<glm::vec4>
{
    static Node encode(const glm::vec4& rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        node.push_back(rhs.w);
        node.SetStyle(EmitterStyle::Flow);
        return node;
    }

    static bool decode(const Node& node, glm::vec4& rhs)
    {
        if(!node.IsSequence() || node.size() != 4) { return false; }

        rhs.x = node[0].as<float>();
        rhs.y = node[1].as<float>();
        rhs.z = node[2].as<float>();
        rhs.w = node[3].as<float>();
        return true;
    }
};

template<>
struct convert<atcg::UUID>
{
    static Node encode(const atcg::UUID& uuid)
    {
        Node node;
        node.push_back((uint64_t)uuid);
        return node;
    }

    static bool decode(const Node& node, atcg::UUID& uuid)
    {
        uuid = node.as<uint64_t>();
        return true;
    }
};

YAML::Emitter& operator<<(YAML::Emitter& out, const glm::vec2& v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << YAML::EndSeq;
    return out;
}

YAML::Emitter& operator<<(YAML::Emitter& out, const glm::vec3& v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
    return out;
}

YAML::Emitter& operator<<(YAML::Emitter& out, const glm::vec4& v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

}    // namespace YAML

namespace atcg
{

namespace detail
{
#define TRANSFORM_COMPONENT_NAME "Transform"
#define TRANSFORM_POSITION_NAME  "Translation"
#define TRANSFORM_SCALE_NAME     "Scale"
#define TRANSFORM_ROTATION_NAME  "EulerAngles"

#define GEOMETRY_COMPONENT_NAME "Geometry"
#define GEOMETRY_TYPE_NAME      "Type"
#define GEOMETRY_VERTICES_NAME  "Vertices"
#define GEOMETRY_EDGES_NAME     "Edges"
#define GEOMETRY_FACES_NAME     "Faces"

#define RENDER_COMPONENT_NAME             "Renderer"
#define RENDER_TYPE_NAME                  "RenderType"
#define RENDER_VERTEX_SHADER_NAME         "VertexShaderPath"
#define RENDER_FRAGMENT_SHADER_NAME       "FragmentShaderPath"
#define RENDER_GEOMETRY_SHADER_NAME       "GeometryShaderPath"
#define RENDER_COLOR_NAME                 "Color"
#define RENDER_POINT_SIZE_NAME            "PointSize"
#define RENDER_INSTANCE_BUFFER_NAME       "Instances"
#define RENDER_RADIUS_NAME                "EdgeRadius"
#define RENDER_MATERIAL_DIFFUSE           "Diffuse"
#define RENDER_MATERIAL_METALLIC          "Metallic"
#define RENDER_MATERIAL_ROUGHNESS         "Roughness"
#define RENDER_MATERIAL_DIFFUSE_TEXTURE   "DiffuseTexture"
#define RENDER_MATERIAL_METALLIC_TEXTURE  "MetallicTexture"
#define RENDER_MATERIAL_ROUGHNESS_TEXTURE "RoughnessTexture"
#define RENDER_MATERIAL_NORMAL_TEXTURE    "NormalTexture"

#define CAMERA_COMPONENT_NAME     "PerspectiveCamera"
#define EDITOR_CAM_COMPONENT_NAME "EditorCamera"
#define CAMERA_POSITION_NAME      "Translation"
#define CAMERA_LOOKAT_NAME        "LookAt"
#define CAMERA_ASPECT_RATIO_NAME  "AspectRatio"
#define CAMERA_FOV_NAME           "FOV"

void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size)
{
    std::ofstream summary_file(file_name, std::ios::out | std::ios::binary);
    summary_file.write(data, byte_size);
    summary_file.close();
}

template<typename T>
std::vector<T> deserializeBuffer(const std::string& file_name)
{
    std::ifstream summary_file(file_name, std::ios::in | std::ios::binary);
    std::vector<char> buffer_char(std::istreambuf_iterator<char>(summary_file), {});
    summary_file.close();

    std::vector<T> buffer(buffer_char.size() / sizeof(T));

    T* data = reinterpret_cast<T*>(buffer_char.data());
    for(uint32_t i = 0; i < buffer_char.size() / sizeof(T); ++i) { buffer[i] = data[i]; }

    return buffer;
}

void serializeTexture(const atcg::ref_ptr<Texture2D>& texture, std::string& path, float gamma = 1.0f)
{
    uint32_t channels = texture->getSpecification().format == TextureFormat::RGBA ||
                                texture->getSpecification().format == TextureFormat::RGBAFLOAT
                            ? 4
                            : 1;
    bool hdr          = texture->getSpecification().format == TextureFormat::RGBAFLOAT ||
               texture->getSpecification().format == TextureFormat::RFLOAT;
    std::string file_ending = hdr ? ".hdr" : ".png";

    auto img_data = texture->getData(atcg::CPU);
    atcg::ref_ptr<Image> img;
    if(hdr) { img = atcg::make_ref<Image>((float*)img_data.data_ptr(), texture->width(), texture->height(), channels); }
    else { img = atcg::make_ref<Image>((uint8_t*)img_data.data_ptr(), texture->width(), texture->height(), channels); }

    path = path + file_ending;
    IO::imwrite(img, path, gamma);
}

void serializeMaterial(YAML::Emitter& out, Entity entity, const Material& material, const std::string& file_path)
{
    out << YAML::Key << "Material";
    out << YAML::BeginMap;

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

        detail::serializeTexture(diffuse_texture, img_path, 1.0f / 2.2f);

        out << YAML::Key << RENDER_MATERIAL_DIFFUSE_TEXTURE << YAML::Value << img_path;
    }
    else
    {
        auto data         = diffuse_texture->getData(atcg::CPU);
        glm::u8vec3 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>()};

        glm::vec3 c(color);
        c = c / 255.0f;

        out << YAML::Key << RENDER_MATERIAL_DIFFUSE << YAML::Value << c;
    }

    if(use_normal_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_normal";

        detail::serializeTexture(normal_texture, img_path);

        out << YAML::Key << RENDER_MATERIAL_NORMAL_TEXTURE << YAML::Value << img_path;
    }

    if(use_metallic_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_metallic";

        detail::serializeTexture(metallic_texture, img_path);

        out << YAML::Key << RENDER_MATERIAL_METALLIC_TEXTURE << YAML::Value << img_path;
    }
    else
    {
        auto data   = metallic_texture->getData(atcg::CPU);
        float color = data.item<float>();

        out << YAML::Key << RENDER_MATERIAL_METALLIC << YAML::Value << color;
    }

    if(use_roughness_texture)
    {
        std::string img_path = file_path + "_" + std::to_string(entity_id) + "_roughness";

        detail::serializeTexture(roughness_texture, img_path, 1.0f / 2.2f);

        out << YAML::Key << RENDER_MATERIAL_ROUGHNESS_TEXTURE << YAML::Value << img_path;
    }
    else
    {
        auto data   = roughness_texture->getData(atcg::CPU);
        float color = data.item<float>();

        out << YAML::Key << RENDER_MATERIAL_ROUGHNESS << YAML::Value << color;
    }

    out << YAML::EndMap;
}

Material deserializeMaterial(const YAML::Node& material_node)
{
    Material material;

    // Diffuse
    if(material_node[RENDER_MATERIAL_DIFFUSE])
    {
        glm::vec3 diffuse_color = material_node[RENDER_MATERIAL_DIFFUSE].as<glm::vec3>();
        material.setDiffuseColor(glm::vec4(diffuse_color, 1.0f));
    }
    else
    {
        std::string diffuse_path = material_node[RENDER_MATERIAL_DIFFUSE_TEXTURE].as<std::string>();
        ATCG_TRACE(diffuse_path);
        auto img             = IO::imread(diffuse_path, 2.2f);
        auto diffuse_texture = atcg::Texture2D::create(img);
        material.setDiffuseTexture(diffuse_texture);
    }

    // Normals
    if(material_node[RENDER_MATERIAL_NORMAL_TEXTURE])
    {
        std::string normal_path = material_node[RENDER_MATERIAL_NORMAL_TEXTURE].as<std::string>();
        auto img                = IO::imread(normal_path);
        auto normal_texture     = atcg::Texture2D::create(img);
        material.setNormalTexture(normal_texture);
    }

    // Roughness
    if(material_node[RENDER_MATERIAL_ROUGHNESS])
    {
        float roughness = material_node[RENDER_MATERIAL_ROUGHNESS].as<float>();
        material.setRoughness(roughness);
    }
    else
    {
        std::string roughness_path = material_node[RENDER_MATERIAL_ROUGHNESS_TEXTURE].as<std::string>();
        auto img                   = IO::imread(roughness_path);
        auto roughness_texture     = atcg::Texture2D::create(img);
        material.setRoughnessTexture(roughness_texture);
    }

    // Metallic
    if(material_node[RENDER_MATERIAL_METALLIC])
    {
        float metallic = material_node[RENDER_MATERIAL_METALLIC].as<float>();
        material.setMetallic(metallic);
    }
    else
    {
        std::string metallic_path = material_node[RENDER_MATERIAL_METALLIC_TEXTURE].as<std::string>();
        auto img                  = IO::imread(metallic_path);
        auto metallic_texture     = atcg::Texture2D::create(img);
        material.setMetallicTexture(metallic_texture);
    }

    return material;
}

void serializeEntity(YAML::Emitter& out, Entity entity, const std::string& file_path)
{
    // Every entity should have these components
    IDComponent id     = entity.getComponent<IDComponent>();
    NameComponent name = entity.getComponent<NameComponent>();
    out << YAML::BeginMap;
    out << YAML::Key << "Entity" << YAML::Value << id.ID;
    out << YAML::Key << "Name" << YAML::Value << name.name;

    if(entity.hasComponent<TransformComponent>())
    {
        out << YAML::Key << TRANSFORM_COMPONENT_NAME;
        out << YAML::BeginMap;

        auto& transform = entity.getComponent<TransformComponent>();
        out << YAML::Key << TRANSFORM_POSITION_NAME << YAML::Value << transform.getPosition();
        out << YAML::Key << TRANSFORM_SCALE_NAME << YAML::Value << transform.getScale();
        out << YAML::Key << TRANSFORM_ROTATION_NAME << YAML::Value << transform.getRotation();

        out << YAML::EndMap;
    }

    if(entity.hasComponent<CameraComponent>())
    {
        out << YAML::Key << CAMERA_COMPONENT_NAME;
        out << YAML::BeginMap;

        auto& camera                         = entity.getComponent<CameraComponent>();
        atcg::ref_ptr<PerspectiveCamera> cam = camera.camera;

        // * The camera component should always be used together with a transform.
        // out << YAML::Key << CAMERA_POSITION_NAME << YAML::Value << cam->getPosition();
        // out << YAML::Key << CAMERA_LOOKAT_NAME << YAML::Value << cam->getLookAt();
        out << YAML::Key << CAMERA_ASPECT_RATIO_NAME << YAML::Value << cam->getAspectRatio();
        out << YAML::Key << CAMERA_FOV_NAME << YAML::Value << cam->getFOV();

        out << YAML::EndMap;
    }

    if(entity.hasComponent<EditorCameraComponent>())
    {
        out << YAML::Key << EDITOR_CAM_COMPONENT_NAME;
        out << YAML::BeginMap;

        auto& camera                         = entity.getComponent<EditorCameraComponent>();
        atcg::ref_ptr<PerspectiveCamera> cam = camera.camera;

        // ? Should the editor camera also hold a transform component? For now it stores position and lookat
        out << YAML::Key << CAMERA_POSITION_NAME << YAML::Value << cam->getPosition();
        out << YAML::Key << CAMERA_LOOKAT_NAME << YAML::Value << cam->getLookAt();
        out << YAML::Key << CAMERA_ASPECT_RATIO_NAME << YAML::Value << cam->getAspectRatio();
        out << YAML::Key << CAMERA_FOV_NAME << YAML::Value << cam->getFOV();

        out << YAML::EndMap;
    }

    if(entity.hasComponent<GeometryComponent>())
    {
        out << YAML::Key << GEOMETRY_COMPONENT_NAME;
        out << YAML::BeginMap;

        auto& geometry             = entity.getComponent<GeometryComponent>();
        atcg::ref_ptr<Graph> graph = geometry.graph;

        out << YAML::Key << GEOMETRY_TYPE_NAME << YAML::Value << (int)graph->type();

        if(graph->n_vertices() != 0)
        {
            const char* buffer      = graph->getVerticesBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".vertices";
            serializeBuffer(buffer_name, buffer, graph->getVerticesBuffer()->size());
            out << YAML::Key << GEOMETRY_VERTICES_NAME << YAML::Value << buffer_name;
            graph->getVerticesBuffer()->unmapHostPointers();
        }

        if(graph->n_faces() != 0)
        {
            const char* buffer      = graph->getFaceIndexBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".faces";
            serializeBuffer(buffer_name, buffer, graph->getFaceIndexBuffer()->size());
            out << YAML::Key << GEOMETRY_FACES_NAME << YAML::Value << buffer_name;
            graph->getFaceIndexBuffer()->unmapHostPointers();
        }

        if(graph->n_edges() != 0)
        {
            const char* buffer      = graph->getEdgesBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".edges";
            serializeBuffer(buffer_name, buffer, graph->getEdgesBuffer()->size());
            out << YAML::Key << GEOMETRY_EDGES_NAME << YAML::Value << buffer_name;
            graph->getEdgesBuffer()->unmapHostPointers();
        }

        out << YAML::EndMap;
    }

    if(!entity.hasAnyComponent<MeshRenderComponent,
                               PointRenderComponent,
                               PointSphereRenderComponent,
                               EdgeRenderComponent,
                               EdgeCylinderRenderComponent,
                               InstanceRenderComponent>())
    {
        out << YAML::EndMap;
        return;
    }

    out << YAML::Key << RENDER_COMPONENT_NAME;
    out << YAML::BeginSeq;

    if(entity.hasComponent<MeshRenderComponent>())
    {
        out << YAML::BeginMap;

        auto& renderer = entity.getComponent<MeshRenderComponent>();

        out << YAML::Key << RENDER_TYPE_NAME << YAML::Value << renderer.draw_mode;
        out << YAML::Key << RENDER_VERTEX_SHADER_NAME << YAML::Value << renderer.shader->getVertexPath();
        out << YAML::Key << RENDER_FRAGMENT_SHADER_NAME << YAML::Value << renderer.shader->getFragmentPath();
        out << YAML::Key << RENDER_GEOMETRY_SHADER_NAME << YAML::Value << renderer.shader->getGeometryPath();

        detail::serializeMaterial(out, entity, renderer.material, file_path);

        out << YAML::EndMap;
    }

    if(entity.hasComponent<PointRenderComponent>())
    {
        out << YAML::BeginMap;

        auto& renderer = entity.getComponent<PointRenderComponent>();

        out << YAML::Key << RENDER_TYPE_NAME << YAML::Value << renderer.draw_mode;
        out << YAML::Key << RENDER_COLOR_NAME << YAML::Value << renderer.color;
        out << YAML::Key << RENDER_POINT_SIZE_NAME << YAML::Value << renderer.point_size;
        out << YAML::Key << RENDER_VERTEX_SHADER_NAME << YAML::Value << renderer.shader->getVertexPath();
        out << YAML::Key << RENDER_FRAGMENT_SHADER_NAME << YAML::Value << renderer.shader->getFragmentPath();
        out << YAML::Key << RENDER_GEOMETRY_SHADER_NAME << YAML::Value << renderer.shader->getGeometryPath();

        out << YAML::EndMap;
    }

    if(entity.hasComponent<PointSphereRenderComponent>())
    {
        out << YAML::BeginMap;

        auto& renderer = entity.getComponent<PointSphereRenderComponent>();

        out << YAML::Key << RENDER_TYPE_NAME << YAML::Value << renderer.draw_mode;
        out << YAML::Key << RENDER_POINT_SIZE_NAME << YAML::Value << renderer.point_size;
        out << YAML::Key << RENDER_VERTEX_SHADER_NAME << YAML::Value << renderer.shader->getVertexPath();
        out << YAML::Key << RENDER_FRAGMENT_SHADER_NAME << YAML::Value << renderer.shader->getFragmentPath();
        out << YAML::Key << RENDER_GEOMETRY_SHADER_NAME << YAML::Value << renderer.shader->getGeometryPath();

        detail::serializeMaterial(out, entity, renderer.material, file_path);

        out << YAML::EndMap;
    }

    if(entity.hasComponent<EdgeRenderComponent>())
    {
        out << YAML::BeginMap;

        auto& renderer = entity.getComponent<EdgeRenderComponent>();

        out << YAML::Key << RENDER_TYPE_NAME << YAML::Value << renderer.draw_mode;
        out << YAML::Key << RENDER_COLOR_NAME << YAML::Value << renderer.color;

        out << YAML::EndMap;
    }

    if(entity.hasComponent<EdgeCylinderRenderComponent>())
    {
        out << YAML::BeginMap;

        auto& renderer = entity.getComponent<EdgeCylinderRenderComponent>();

        out << YAML::Key << RENDER_TYPE_NAME << YAML::Value << renderer.draw_mode;
        out << YAML::Key << RENDER_RADIUS_NAME << YAML::Value << renderer.radius;

        detail::serializeMaterial(out, entity, renderer.material, file_path);

        out << YAML::EndMap;
    }

    if(entity.hasComponent<InstanceRenderComponent>())
    {
        out << YAML::BeginMap;

        auto& renderer = entity.getComponent<InstanceRenderComponent>();

        out << YAML::Key << RENDER_TYPE_NAME << YAML::Value << renderer.draw_mode;

        const char* buffer      = renderer.instance_vbo->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".instances";
        serializeBuffer(buffer_name, buffer, renderer.instance_vbo->size());
        out << YAML::Key << RENDER_INSTANCE_BUFFER_NAME << YAML::Value << buffer_name;
        renderer.instance_vbo->unmapHostPointers();

        detail::serializeMaterial(out, entity, renderer.material, file_path);

        out << YAML::EndMap;
    }

    out << YAML::EndSeq;
    out << YAML::EndMap;
}

}    // namespace detail

Serializer::Serializer(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

void Serializer::serialize(const std::string& file_path)
{
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "Scene" << YAML::Value << "Untitled";

    out << YAML::Key << "Entities" << YAML::Value << YAML::BeginSeq;

    _scene->_registry.each(
        [&](auto entityID)
        {
            Entity entity = {entityID, _scene.get()};
            if(!entity) return;

            detail::serializeEntity(out, entity, file_path);
        });

    out << YAML::EndSeq;
    out << YAML::EndMap;

    std::ofstream fout(file_path);
    fout << out.c_str();
}

void Serializer::deserialize(const std::string& file_path)
{
    YAML::Node data;

    try
    {
        data = YAML::LoadFile(file_path);
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR("Failed to load scene file '{0}'\n   {1}", file_path, e.what());
        return;
    }

    if(!data["Scene"]) return;

    auto entities = data["Entities"];
    if(entities)
    {
        for(auto entity: entities)
        {
            uint64_t uuid    = entity["Entity"].as<uint64_t>();
            std::string name = entity["Name"].as<std::string>();

            Entity deserializedEntity = _scene->createEntity(name);
            auto& idComponent         = deserializedEntity.getComponent<IDComponent>();
            idComponent.ID            = uuid;

            auto transformComponent = entity[TRANSFORM_COMPONENT_NAME];
            if(transformComponent)
            {
                auto& transform = deserializedEntity.addComponent<TransformComponent>();
                transform.setPosition(transformComponent[TRANSFORM_POSITION_NAME].as<glm::vec3>());
                transform.setScale(transformComponent[TRANSFORM_SCALE_NAME].as<glm::vec3>());
                transform.setRotation(transformComponent[TRANSFORM_ROTATION_NAME].as<glm::vec3>());
            }

            auto cameraComponent = entity[CAMERA_COMPONENT_NAME];
            if(cameraComponent)
            {
                float aspect_ratio = cameraComponent[CAMERA_ASPECT_RATIO_NAME].as<float>();
                float fov          = cameraComponent[CAMERA_FOV_NAME].as<float>();
                auto& camera       = deserializedEntity.addComponent<CameraComponent>(
                    atcg::make_ref<atcg::PerspectiveCamera>(aspect_ratio, glm::vec3(0), glm::vec3(0, 0, 1)));
                atcg::ref_ptr<PerspectiveCamera> cam = camera.camera;
                // TODO: Keep consistent
                cam->setFOV(fov);
                // Should always have a transform, otherwise construct default camera
                if(deserializedEntity.hasComponent<TransformComponent>())
                {
                    cam->setFromTransform(deserializedEntity.getComponent<TransformComponent>().getModel());
                }
            }

            auto editorCameraComponent = entity[EDITOR_CAM_COMPONENT_NAME];
            if(editorCameraComponent)
            {
                glm::vec3 position = editorCameraComponent[CAMERA_POSITION_NAME].as<glm::vec3>();
                glm::vec3 lookat   = editorCameraComponent[CAMERA_LOOKAT_NAME].as<glm::vec3>();
                float aspect_ratio = editorCameraComponent[CAMERA_ASPECT_RATIO_NAME].as<float>();
                float fov          = cameraComponent[CAMERA_FOV_NAME].as<float>();
                auto& camera       = deserializedEntity.addComponent<EditorCameraComponent>(
                    atcg::make_ref<atcg::PerspectiveCamera>(aspect_ratio, position, lookat));
                atcg::ref_ptr<PerspectiveCamera> cam = camera.camera;
                cam->setFOV(fov);
            }

            auto geometryComponent = entity[GEOMETRY_COMPONENT_NAME];
            if(geometryComponent)
            {
                auto& geometry       = deserializedEntity.addComponent<GeometryComponent>();
                atcg::GraphType type = (atcg::GraphType)geometryComponent[GEOMETRY_TYPE_NAME].as<int>();

                switch(type)
                {
                    case atcg::GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH:
                    {
                        std::string vertex_path         = geometryComponent[GEOMETRY_VERTICES_NAME].as<std::string>();
                        std::string faces_path          = geometryComponent[GEOMETRY_FACES_NAME].as<std::string>();
                        std::vector<Vertex> vertices    = detail::deserializeBuffer<Vertex>(vertex_path);
                        std::vector<glm::u32vec3> faces = detail::deserializeBuffer<glm::u32vec3>(faces_path);

                        geometry.graph = Graph::createTriangleMesh(vertices, faces);
                    }
                    break;
                    case atcg::GraphType::ATCG_GRAPH_TYPE_POINTCLOUD:
                    {
                        std::string vertex_path      = geometryComponent[GEOMETRY_VERTICES_NAME].as<std::string>();
                        std::vector<Vertex> vertices = detail::deserializeBuffer<Vertex>(vertex_path);

                        geometry.graph = Graph::createPointCloud(vertices);
                    }
                    break;
                    case atcg::GraphType::ATCG_GRAPH_TYPE_GRAPH:
                    {
                        std::string vertex_path      = geometryComponent[GEOMETRY_VERTICES_NAME].as<std::string>();
                        std::string edges_path       = geometryComponent[GEOMETRY_EDGES_NAME].as<std::string>();
                        std::vector<Vertex> vertices = detail::deserializeBuffer<Vertex>(vertex_path);
                        std::vector<Edge> edges      = detail::deserializeBuffer<Edge>(edges_path);

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

            auto renderers = entity[RENDER_COMPONENT_NAME];
            if(!renderers) continue;

            for(auto renderer: renderers)
            {
                atcg::DrawMode mode = (atcg::DrawMode)renderer[RENDER_TYPE_NAME].as<int>();

                switch(mode)
                {
                    case atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE:
                    {
                        auto& renderComponent     = deserializedEntity.addComponent<MeshRenderComponent>();
                        renderComponent.draw_mode = mode;
                        std::string vertex_path   = renderer[RENDER_VERTEX_SHADER_NAME].as<std::string>();
                        std::string fragment_path = renderer[RENDER_FRAGMENT_SHADER_NAME].as<std::string>();
                        std::string geometry_path = renderer[RENDER_GEOMETRY_SHADER_NAME].as<std::string>();

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
                        else { renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path); }

                        auto material_node = renderer["Material"];

                        if(material_node)
                        {
                            Material material        = detail::deserializeMaterial(material_node);
                            renderComponent.material = material;
                        }
                    }
                    break;
                    case atcg::DrawMode::ATCG_DRAW_MODE_POINTS:
                    {
                        auto& renderComponent      = deserializedEntity.addComponent<PointRenderComponent>();
                        renderComponent.draw_mode  = mode;
                        renderComponent.color      = renderer[RENDER_COLOR_NAME].as<glm::vec3>();
                        renderComponent.point_size = renderer[RENDER_POINT_SIZE_NAME].as<float>();
                        std::string vertex_path    = renderer[RENDER_VERTEX_SHADER_NAME].as<std::string>();
                        std::string fragment_path  = renderer[RENDER_FRAGMENT_SHADER_NAME].as<std::string>();
                        std::string geometry_path  = renderer[RENDER_GEOMETRY_SHADER_NAME].as<std::string>();

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
                        else { renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path); }
                    }
                    break;
                    case atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE:
                    {
                        auto& renderComponent      = deserializedEntity.addComponent<PointSphereRenderComponent>();
                        renderComponent.draw_mode  = mode;
                        renderComponent.point_size = renderer[RENDER_POINT_SIZE_NAME].as<float>();
                        std::string vertex_path    = renderer[RENDER_VERTEX_SHADER_NAME].as<std::string>();
                        std::string fragment_path  = renderer[RENDER_FRAGMENT_SHADER_NAME].as<std::string>();
                        std::string geometry_path  = renderer[RENDER_GEOMETRY_SHADER_NAME].as<std::string>();

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
                        else { renderComponent.shader = atcg::make_ref<Shader>(vertex_path, fragment_path); }

                        auto material_node = renderer["Material"];

                        if(material_node)
                        {
                            Material material        = detail::deserializeMaterial(material_node);
                            renderComponent.material = material;
                        }
                    }
                    break;
                    case atcg::DrawMode::ATCG_DRAW_MODE_EDGES:
                    {
                        auto& renderComponent     = deserializedEntity.addComponent<EdgeRenderComponent>();
                        renderComponent.draw_mode = mode;
                        renderComponent.color     = renderer[RENDER_COLOR_NAME].as<glm::vec3>();
                    }
                    break;
                    case atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER:
                    {
                        auto& renderComponent     = deserializedEntity.addComponent<EdgeCylinderRenderComponent>();
                        renderComponent.draw_mode = mode;
                        renderComponent.radius    = renderer[RENDER_RADIUS_NAME].as<float>();

                        auto material_node = renderer["Material"];

                        if(material_node)
                        {
                            Material material        = detail::deserializeMaterial(material_node);
                            renderComponent.material = material;
                        }
                    }
                    break;
                    case atcg::DrawMode::ATCG_DRAW_MODE_INSTANCED:
                    {
                        std::string buffer_path      = renderer[RENDER_INSTANCE_BUFFER_NAME].as<std::string>();
                        std::vector<Instance> buffer = detail::deserializeBuffer<Instance>(buffer_path);

                        auto& renderComponent     = deserializedEntity.addComponent<InstanceRenderComponent>(buffer);
                        renderComponent.draw_mode = mode;

                        auto material_node = renderer["Material"];

                        if(material_node)
                        {
                            Material material        = detail::deserializeMaterial(material_node);
                            renderComponent.material = material;
                        }
                    }
                    break;
                }
            }
        }
    }
}
}    // namespace atcg