#include <Asset/AssetExporter.h>

#include <DataStructure/Image.h>
#include <Renderer/Texture.h>
#include <Renderer/Material.h>
#include <DataStructure/Graph.h>
#include <Scripting/Script.h>
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

ATCG_INLINE std::string
serialize_texture_ver1(const atcg::ref_ptr<Texture2D>& texture, const std::filesystem::path& path, float gamma = 1.0f)
{
    torch::Tensor texture_data = texture->getData(atcg::CPU);

    Image img(texture_data);

    std::string file_ending = ".png";
    if(img.isHDR())
    {
        file_ending = ".hdr";
    }

    auto img_path = path.string() + file_ending;
    img.applyGamma(gamma);
    img.store(img_path);

    return file_ending;
}

ATCG_INLINE void serialize_material_ver1(const std::filesystem::path& path, const atcg::ref_ptr<Material>& material)
{
    nlohmann::json material_json;

    material_json["Version"] = "1.0";

    std::vector<nlohmann::json> serialized_registry;

    auto diffuse_texture   = material->getDiffuseTexture();
    auto normal_texture    = material->getNormalTexture();
    auto metallic_texture  = material->getMetallicTexture();
    auto roughness_texture = material->getRoughnessTexture();

    bool use_diffuse_texture   = !(diffuse_texture->width() == 1 && diffuse_texture->height() == 1);
    bool use_normal_texture    = !(normal_texture->width() == 1 && normal_texture->height() == 1);
    bool use_metallic_texture  = !(metallic_texture->width() == 1 && metallic_texture->height() == 1);
    bool use_roughness_texture = !(roughness_texture->width() == 1 && roughness_texture->height() == 1);

    if(use_diffuse_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "diffuse";

        auto file_ending = serialize_texture_ver1(diffuse_texture, img_path, 1.0f / 2.2f);

        material_json[DIFFUSE_TEXTURE_KEY] = img_path.generic_string() + file_ending;
    }
    else
    {
        auto data         = diffuse_texture->getData(atcg::CPU);
        glm::u8vec3 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>()};

        glm::vec3 c(color);
        c = c / 255.0f;

        material_json[DIFFUSE_KEY] = nlohmann::json::array({c.x, c.y, c.z});
    }

    if(use_normal_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "normals";

        auto file_ending = serialize_texture_ver1(normal_texture, img_path);

        material_json[NORMAL_TEXTURE_KEY] = img_path.generic_string() + file_ending;
    }

    if(use_metallic_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "metallic";

        auto file_ending = serialize_texture_ver1(metallic_texture, img_path);

        material_json[METALLIC_TEXTURE_KEY] = img_path.generic_string() + file_ending;
    }
    else
    {
        auto data   = metallic_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_json[METALLIC_KEY] = color;
    }

    if(use_roughness_texture)
    {
        std::filesystem::path img_path = path.parent_path() / "roughness";

        auto file_ending = serialize_texture_ver1(roughness_texture, img_path);

        material_json[ROUGHNESS_TEXTURE_KEY] = img_path.generic_string() + file_ending;
    }
    else
    {
        auto data   = roughness_texture->getData(atcg::CPU);
        float color = data.item<float>();

        material_json[ROUGHNESS_KEY] = color;
    }

    std::ofstream o(path);
    o << std::setw(4) << material_json << std::endl;
}

ATCG_INLINE void
serialize_buffer_ver1(const std::filesystem::path& file_name, const char* data, const uint32_t byte_size)
{
    std::ofstream summary_file(file_name, std::ios::out | std::ios::binary);
    summary_file.write(data, byte_size);
    summary_file.close();
}

ATCG_INLINE void serialize_graph_ver1(const atcg::ref_ptr<Graph>& graph, const std::filesystem::path& path)
{
    nlohmann::json graph_json;

    graph_json["Version"] = "1.0";

    graph_json[TYPE_KEY] = graphTypeToString(graph->type());

    if(graph->n_vertices() != 0)
    {
        const char* buffer = graph->getVerticesBuffer()->getHostPointer<char>();
        auto buffer_name   = path.parent_path() / "vertices.bin";
        serialize_buffer_ver1(buffer_name, buffer, graph->getVerticesBuffer()->size());
        graph_json[VERTICES_KEY] = buffer_name;
        graph->getVerticesBuffer()->unmapHostPointers();
    }

    if(graph->n_faces() != 0)
    {
        const char* buffer = graph->getFaceIndexBuffer()->getHostPointer<char>();
        auto buffer_name   = path.parent_path() / "faces.bin";
        serialize_buffer_ver1(buffer_name, buffer, graph->getFaceIndexBuffer()->size());
        graph_json[FACES_KEY] = buffer_name;
        graph->getFaceIndexBuffer()->unmapHostPointers();
    }

    if(graph->n_edges() != 0)
    {
        const char* buffer = graph->getEdgesBuffer()->getHostPointer<char>();
        auto buffer_name   = path.parent_path() / "edges.bin";
        serialize_buffer_ver1(buffer_name, buffer, graph->getEdgesBuffer()->size());
        graph_json[EDGES_KEY] = buffer_name;
        graph->getEdgesBuffer()->unmapHostPointers();
    }

    std::ofstream o(path);
    o << std::setw(4) << graph_json << std::endl;
}

ATCG_INLINE void serialize_script_ver1(const atcg::ref_ptr<Script>& script, const std::filesystem::path& path)
{
    auto path_ = path;
    std::filesystem::copy(script->getFilePath(), path_.replace_extension(".py"));
}

ATCG_INLINE void
export_asset_ver1(const std::filesystem::path& path, const atcg::ref_ptr<Asset>& asset, const AssetMetaData& data)
{
    switch(asset->getType())
    {
        case AssetType::Graph:
        {
            auto graph_path = path / "graphs" / std::to_string(asset->handle);
            std::filesystem::create_directories(graph_path);
            serialize_graph_ver1(std::dynamic_pointer_cast<Graph>(asset), graph_path / (data.name + ".graph"));
        }
        break;
        case AssetType::Material:
        {
            auto material_path = path / "materials" / std::to_string(asset->handle);
            std::filesystem::create_directories(material_path);
            serialize_material_ver1(material_path / (data.name + ".mat"),
                                    std::dynamic_pointer_cast<atcg::Material>(asset));
        }
        break;
        case AssetType::Texture2D:
        {
            auto texture_path = path / "textures" / std::to_string(asset->handle);
            std::filesystem::create_directories(texture_path);
            serialize_texture_ver1(std::dynamic_pointer_cast<atcg::Texture2D>(asset), texture_path / data.name);
        }
        case AssetType::Scene:
        {
            // TODO: Templates ?
            auto scene_path = path / "scenes" / std::to_string(asset->handle);
            std::filesystem::create_directories(scene_path);
            atcg::Serialization::SceneSerializer serializer(std::dynamic_pointer_cast<atcg::Scene>(asset));
            serializer.serialize((scene_path / data.name).replace_extension(".scene").string());
        }
        break;
        case AssetType::Script:
        {
            auto script_path = path / "scripts" / std::to_string(asset->handle);
            std::filesystem::create_directories(script_path);
            serialize_script_ver1(std::dynamic_pointer_cast<Script>(asset), script_path / data.name);
        }
        break;
    }
}
}    // namespace detail

void AssetExporter::exportAsset(const std::filesystem::path& path,
                                const atcg::ref_ptr<Asset>& asset,
                                const AssetMetaData& data)
{
    detail::export_asset_ver1(path, asset, data);
}
}    // namespace atcg