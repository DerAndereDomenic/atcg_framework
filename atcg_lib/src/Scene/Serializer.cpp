#include <Scene/Serializer.h>

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

        if(graph->getVerticesBuffer())
        {
            const char* buffer      = graph->getVerticesBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + ".vertices";
            serializeBuffer(buffer_name, buffer, graph->getVerticesBuffer()->size());
            out << YAML::Key << GEOMETRY_VERTICES_NAME << YAML::Value << buffer_name;
            graph->getVerticesBuffer()->unmapHostPointers();
        }

        if(graph->getFaceIndexBuffer())
        {
            const char* buffer      = graph->getFaceIndexBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + ".faces";
            serializeBuffer(buffer_name, buffer, graph->getFaceIndexBuffer()->size());
            out << YAML::Key << GEOMETRY_FACES_NAME << YAML::Value << buffer_name;
            graph->getFaceIndexBuffer()->unmapHostPointers();
        }

        if(graph->getEdgesBuffer())
        {
            const char* buffer      = graph->getEdgesBuffer()->getHostPointer<char>();
            std::string buffer_name = file_path + ".edges";
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

void Serializer::deserialize(const std::string& file_path) {}
}    // namespace atcg