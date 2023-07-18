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
#define GEOMETRY_RADIUS_NAME    "EdgeRadius"

#define RENDER_COMPONENT_NAME       "Renderer"
#define RENDER_TYPE_NAME            "RenderType"
#define RENDER_VERTEX_SHADER_NAME   "VertexShaderPath"
#define RENDER_FRAGMENT_SHADER_NAME "FragmentShaderPath"
#define RENDER_GEOMETRY_SHADER_NAME "GeometryShaderPath"
#define RENDER_COLOR_NAME           "Color"

#define CAMERA_COMPONENT_NAME "PerspectiveCamera"
#define CAMERA_POSITION_NAME  "Translation"
#define CAMERA_LOOKAT_NAME    "LookAt"

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

void serializeEntity(YAML::Emitter& out, Entity entity, const std::string& file_path)
{
    IDComponent id = entity.getComponent<IDComponent>();
    out << YAML::BeginMap;
    out << YAML::Key << "Entity" << YAML::Value << id.ID;

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

        out << YAML::Key << CAMERA_POSITION_NAME << YAML::Value << cam->getPosition();
        out << YAML::Key << CAMERA_LOOKAT_NAME << YAML::Value << cam->getLookAt();

        out << YAML::EndMap;
    }

    if(entity.hasComponent<GeometryComponent>())
    {
        out << YAML::Key << GEOMETRY_COMPONENT_NAME;
        out << YAML::BeginMap;

        auto& geometry             = entity.getComponent<GeometryComponent>();
        atcg::ref_ptr<Graph> graph = geometry.graph;

        out << YAML::Key << GEOMETRY_TYPE_NAME << YAML::Value << (int)graph->type();
        out << YAML::Key << GEOMETRY_RADIUS_NAME << YAML::Value << graph->edge_radius();

        if(graph->getVerticesBuffer())
        {
            const char* buffer      = graph->getVerticesBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".vertices";
            serializeBuffer(buffer_name, buffer, graph->getVerticesBuffer()->size());
            out << YAML::Key << GEOMETRY_VERTICES_NAME << YAML::Value << buffer_name;
            graph->getVerticesBuffer()->unmapHostPointers();
        }

        if(graph->getFaceIndexBuffer())
        {
            const char* buffer      = graph->getFaceIndexBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + "." + std::to_string(id.ID) + ".faces";
            serializeBuffer(buffer_name, buffer, graph->getFaceIndexBuffer()->size());
            out << YAML::Key << GEOMETRY_FACES_NAME << YAML::Value << buffer_name;
            graph->getFaceIndexBuffer()->unmapHostPointers();
        }

        if(graph->getEdgesBuffer())
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
                               EdgeCylinderRenderComponent>())
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
        out << YAML::Key << RENDER_COLOR_NAME << YAML::Value << renderer.color;
        out << YAML::Key << RENDER_VERTEX_SHADER_NAME << YAML::Value << renderer.shader->getVertexPath();
        out << YAML::Key << RENDER_FRAGMENT_SHADER_NAME << YAML::Value << renderer.shader->getFragmentPath();
        out << YAML::Key << RENDER_GEOMETRY_SHADER_NAME << YAML::Value << renderer.shader->getGeometryPath();

        out << YAML::EndMap;
    }

    if(entity.hasComponent<PointRenderComponent>())
    {
        out << YAML::BeginMap;

        auto& renderer = entity.getComponent<PointRenderComponent>();

        out << YAML::Key << RENDER_TYPE_NAME << YAML::Value << renderer.draw_mode;
        out << YAML::Key << RENDER_COLOR_NAME << YAML::Value << renderer.color;
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
        out << YAML::Key << RENDER_COLOR_NAME << YAML::Value << renderer.color;
        out << YAML::Key << RENDER_VERTEX_SHADER_NAME << YAML::Value << renderer.shader->getVertexPath();
        out << YAML::Key << RENDER_FRAGMENT_SHADER_NAME << YAML::Value << renderer.shader->getFragmentPath();
        out << YAML::Key << RENDER_GEOMETRY_SHADER_NAME << YAML::Value << renderer.shader->getGeometryPath();

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
        out << YAML::Key << RENDER_COLOR_NAME << YAML::Value << renderer.color;

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
            uint64_t uuid = entity["Entity"].as<uint64_t>();

            Entity deserializedEntity = _scene->createEntity();
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
                glm::vec3 position = cameraComponent[CAMERA_POSITION_NAME].as<glm::vec3>();
                glm::vec3 lookat   = cameraComponent[CAMERA_LOOKAT_NAME].as<glm::vec3>();
                auto& window       = atcg::Application::get()->getWindow();
                float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
                auto& camera       = deserializedEntity.addComponent<CameraComponent>(
                    atcg::make_ref<atcg::PerspectiveCamera>(aspect_ratio, position, lookat));
            }

            auto geometryComponent = entity[GEOMETRY_COMPONENT_NAME];
            if(geometryComponent)
            {
                auto& geometry       = deserializedEntity.addComponent<GeometryComponent>();
                atcg::GraphType type = (atcg::GraphType)geometryComponent[GEOMETRY_TYPE_NAME].as<int>();
                float edge_radius    = geometryComponent[GEOMETRY_RADIUS_NAME].as<float>();

                switch(type)
                {
                    case atcg::GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH:
                    {
                        std::string vertex_path         = geometryComponent[GEOMETRY_VERTICES_NAME].as<std::string>();
                        std::string faces_path          = geometryComponent[GEOMETRY_FACES_NAME].as<std::string>();
                        std::vector<Vertex> vertices    = detail::deserializeBuffer<Vertex>(vertex_path);
                        std::vector<glm::u32vec3> faces = detail::deserializeBuffer<glm::u32vec3>(faces_path);

                        geometry.graph = Graph::createTriangleMesh(vertices, faces, edge_radius);
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
                        renderComponent.color     = renderer[RENDER_COLOR_NAME].as<glm::vec3>();
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
                    }
                    break;
                    case atcg::DrawMode::ATCG_DRAW_MODE_POINTS:
                    {
                        auto& renderComponent     = deserializedEntity.addComponent<PointRenderComponent>();
                        renderComponent.draw_mode = mode;
                        renderComponent.color     = renderer[RENDER_COLOR_NAME].as<glm::vec3>();
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
                    }
                    break;
                    case atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE:
                    {
                        auto& renderComponent     = deserializedEntity.addComponent<PointSphereRenderComponent>();
                        renderComponent.draw_mode = mode;
                        renderComponent.color     = renderer[RENDER_COLOR_NAME].as<glm::vec3>();
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
                        renderComponent.color     = renderer[RENDER_COLOR_NAME].as<glm::vec3>();
                    }
                    break;
                }
            }
        }
    }
}
}    // namespace atcg