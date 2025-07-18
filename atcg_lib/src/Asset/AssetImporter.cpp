#include <Asset/AssetImporter.h>

#include <DataStructure/Image.h>
#include <Renderer/Texture.h>
#include <Renderer/Material.h>
#include <DataStructure/Graph.h>
#include <Scripting/Script.h>
#include <Renderer/Shader.h>
#include <Scene/Serializer.h>

#include <json.hpp>

namespace atcg
{

namespace detail
{

#define DIFFUSE_KEY           "Diffuse"
#define DIFFUSE_TEXTURE_KEY   "DiffuseTexture"
#define NORMAL_TEXTURE_KEY    "NormalTexture"
#define ROUGHNESS_KEY         "Roughness"
#define ROUGHNESS_TEXTURE_KEY "RoughnessTexture"
#define METALLIC_KEY          "Metallic"
#define METALLIC_TEXTURE_KEY  "MetallicTexture"
#define TYPE_KEY              "Type"
#define VERTICES_KEY          "Vertices"
#define FACES_KEY             "Faces"
#define EDGES_KEY             "Edges"
#define GEOMETRY_KEY          "Geometry"

atcg::ref_ptr<Asset> deserializeMaterial_ver1(const std::filesystem::path& path, const nlohmann::json& material_node)
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
        std::filesystem::path diffuse_path = path.parent_path() / material_node[DIFFUSE_TEXTURE_KEY];
        auto img                           = IO::imread(diffuse_path.generic_string(), 2.2f);
        auto diffuse_texture               = atcg::Texture2D::create(img);
        material->setDiffuseTexture(diffuse_texture);
    }

    // Normals
    if(material_node.contains(NORMAL_TEXTURE_KEY))
    {
        std::filesystem::path normal_path = path.parent_path() / material_node[NORMAL_TEXTURE_KEY];
        auto img                          = IO::imread(normal_path.generic_string());
        auto normal_texture               = atcg::Texture2D::create(img);
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
        std::filesystem::path roughness_path = path.parent_path() / material_node[ROUGHNESS_TEXTURE_KEY];
        auto img                             = IO::imread(roughness_path.generic_string());
        auto roughness_texture               = atcg::Texture2D::create(img);
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
        std::filesystem::path metallic_path = path.parent_path() / material_node[METALLIC_TEXTURE_KEY];
        auto img                            = IO::imread(metallic_path.generic_string());
        auto metallic_texture               = atcg::Texture2D::create(img);
        material->setMetallicTexture(metallic_texture);
    }

    return material;
}

std::vector<uint8_t> deserializeBuffer_ver1(const std::filesystem::path& file_name)
{
    std::ifstream summary_file(file_name, std::ios::in | std::ios::binary);
    std::vector<uint8_t> buffer_char(std::istreambuf_iterator<char>(summary_file), {});
    summary_file.close();

    return buffer_char;
}

atcg::ref_ptr<Asset> deserializeGraph_ver1(const std::filesystem::path& path, const nlohmann::json& j)
{
    atcg::ref_ptr<Asset> asset = nullptr;

    atcg::GraphType type = stringToGraphType(j[TYPE_KEY]);

    switch(type)
    {
        case atcg::GraphType::ATCG_GRAPH_TYPE_TRIANGLEMESH:
        {
            std::filesystem::path vertex_path = path.parent_path() / j[VERTICES_KEY];
            std::filesystem::path faces_path  = path.parent_path() / j[FACES_KEY];
            std::vector<uint8_t> vertices_raw = deserializeBuffer_ver1(vertex_path);
            std::vector<uint8_t> faces_raw    = deserializeBuffer_ver1(faces_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            auto faces = atcg::createHostTensorFromPointer((int32_t*)faces_raw.data(),
                                                           {(int)(faces_raw.size() / sizeof(glm::u32vec3)), 3});

            asset = Graph::createTriangleMesh(vertices, faces);
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_POINTCLOUD:
        {
            std::filesystem::path vertex_path = path.parent_path() / j[VERTICES_KEY];
            std::vector<uint8_t> vertices_raw = deserializeBuffer_ver1(vertex_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            asset = Graph::createPointCloud(vertices);
        }
        break;
        case atcg::GraphType::ATCG_GRAPH_TYPE_GRAPH:
        {
            std::filesystem::path vertex_path = path.parent_path() / j[VERTICES_KEY];
            std::filesystem::path edges_path  = path.parent_path() / j[EDGES_KEY];
            std::vector<uint8_t> vertices_raw = deserializeBuffer_ver1(vertex_path);
            std::vector<uint8_t> edges_raw    = deserializeBuffer_ver1(edges_path);

            auto vertices = atcg::createHostTensorFromPointer(
                (float*)vertices_raw.data(),
                {(int)(vertices_raw.size() / sizeof(Vertex)), atcg::VertexSpecification::VERTEX_SIZE});

            auto edges = atcg::createHostTensorFromPointer(
                (float*)edges_raw.data(),
                {(int)(edges_raw.size() / sizeof(Edge)), atcg::EdgeSpecification::EDGE_SIZE});

            asset = Graph::createGraph(vertices, edges);
        }
        break;
        default:
        {
            ATCG_ERROR("Unknown graph type: {0}", (int)type);
        }
        break;
    }

    return asset;
}

atcg::ref_ptr<Asset> deserializeScript_ver1(const std::filesystem::path& path)
{
    auto script = atcg::make_ref<PythonScript>(path);
    script->init();

    return script;
}

atcg::ref_ptr<Asset> deserializeShader_ver1(const std::filesystem::path& path, const nlohmann::json& j)
{
    atcg::ref_ptr<Asset> asset = nullptr;
    if(j.contains("Compute"))
    {
        auto compute_path = path.parent_path() / j["Compute"];

        asset = atcg::make_ref<Shader>(compute_path.generic_string());
    }
    else
    {
        auto vertex_path   = path.parent_path() / j["Vertex"];
        auto fragment_path = path.parent_path() / j["Fragment"];

        if(j.contains("Geometry"))
        {
            auto geometry_path = path.parent_path() / j["Geometry"];
            asset              = atcg::make_ref<Shader>(vertex_path.generic_string(),
                                           fragment_path.generic_string(),
                                           geometry_path.generic_string());
        }
        else
        {
            asset = atcg::make_ref<Shader>(vertex_path.generic_string(), fragment_path.generic_string());
        }
    }

    return asset;
}

atcg::ref_ptr<Asset> deserializeTexture2D_ver1(const std::filesystem::path& path)
{
    auto img = atcg::IO::imread(path.generic_string());

    return atcg::Texture2D::create(img);
}

atcg::ref_ptr<Asset> deserializeScene_ver1(const std::filesystem::path& path)
{
    auto scene = atcg::make_ref<Scene>();
    atcg::Serialization::SceneSerializer serializer(scene);
    serializer.deserialize(path.generic_string());

    return scene;
}
}    // namespace detail

atcg::ref_ptr<Asset>
AssetImporter::importAsset(const std::filesystem::path& path, AssetHandle handle, const AssetMetaData& metadata)
{
    atcg::ref_ptr<Asset> asset = nullptr;
    switch(metadata.type)
    {
        case AssetType::Scene:
        {
            auto scene_path = path / "scenes" / std::to_string(handle) / (metadata.name + ".scene");

            if(!std::filesystem::exists(scene_path)) break;

            std::ifstream i(scene_path);
            nlohmann::json j;
            i >> j;

            std::string version = j["Version"];

            if(version == "1.0")
            {
                asset = detail::deserializeScene_ver1(scene_path);
            }
        }
        break;
        case AssetType::Texture2D:
        {
            auto img_path = path / "textures" / std::to_string(handle);

            if(!std::filesystem::exists(img_path)) break;

            for(const auto& entry: std::filesystem::directory_iterator(img_path))
            {
                if(entry.is_regular_file())
                {
                    if(entry.path().stem() == metadata.name)
                    {
                        img_path = img_path / entry.path().filename();
                        break;
                    }
                }
            }

            if(std::filesystem::exists(img_path)) asset = detail::deserializeTexture2D_ver1(img_path);
        }
        break;
        case AssetType::Material:
        {
            auto material_path = path / "materials" / std::to_string(handle) / (metadata.name + ".mat");
            if(!std::filesystem::exists(material_path)) break;

            std::ifstream i(material_path);
            nlohmann::json j;
            i >> j;

            std::string version = j["Version"];

            if(version == "1.0")
            {
                asset = detail::deserializeMaterial_ver1(material_path, j);
            }
        }
        break;
        case AssetType::Graph:
        {
            auto graph_path = path / "graphs" / std::to_string(handle) / (metadata.name + ".graph");
            if(!std::filesystem::exists(graph_path)) break;

            std::ifstream i(graph_path);
            nlohmann::json j;
            i >> j;

            std::string version = j["Version"];

            if(version == "1.0")
            {
                asset = detail::deserializeGraph_ver1(graph_path, j);
            }
        }
        break;
        case AssetType::Script:
        {
            auto script_path = path / "scripts" / std::to_string(handle) / (metadata.name + ".py");
            if(std::filesystem::exists(script_path)) asset = detail::deserializeScript_ver1(script_path);
        }
        break;
        case AssetType::Shader:
        {
            auto shader_path = path / "shader" / std::to_string(handle) / (metadata.name + ".json");
            if(!std::filesystem::exists(shader_path)) break;

            std::ifstream i(shader_path);
            nlohmann::json j;
            i >> j;

            std::string version = j["Version"];

            if(version == "1.0")
            {
                asset = detail::deserializeShader_ver1(shader_path, j);
            }
        }
        break;
    }

    return asset;
}
}    // namespace atcg