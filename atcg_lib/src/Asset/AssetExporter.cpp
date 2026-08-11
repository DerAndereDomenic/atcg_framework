#include <Asset/AssetExporter.h>

#include <DataStructure/Image.h>
#include <Renderer/Texture.h>
#include <Material/Material.h>
#include <Renderer/Shader.h>
#include <DataStructure/Graph.h>
#include <Scripting/Script.h>
#include <Scene/Serializer.h>

#include <fstream>

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
#define IOR_KEY               "IoR"
#define IOR_TEXTURE_KEY       "IoRTexture"
#define TYPE_KEY              "Type"
#define VERTICES_KEY          "Vertices"
#define FACES_KEY             "Faces"
#define EDGES_KEY             "Edges"
#define GEOMETRY_KEY          "Geometry"

ATCG_INLINE std::string
serialize_texture2d_ver1(const atcg::ref_ptr<Texture2D>& texture, const std::filesystem::path& path, float gamma = 1.0f)
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

ATCG_INLINE void serialize_material_ver1(const atcg::ref_ptr<Material>& material, const std::filesystem::path& path)
{
    const std::string& type = material->getMaterialType();

    MaterialRegistry::serializeMaterial(material, path);
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
        graph_json[VERTICES_KEY] = "vertices.bin";
        graph->getVerticesBuffer()->unmapHostPointers();
    }

    if(graph->n_faces() != 0)
    {
        const char* buffer = graph->getFaceIndexBuffer()->getHostPointer<char>();
        auto buffer_name   = path.parent_path() / "faces.bin";
        serialize_buffer_ver1(buffer_name, buffer, graph->getFaceIndexBuffer()->size());
        graph_json[FACES_KEY] = "faces.bin";
        graph->getFaceIndexBuffer()->unmapHostPointers();
    }

    if(graph->n_edges() != 0)
    {
        const char* buffer = graph->getEdgesBuffer()->getHostPointer<char>();
        auto buffer_name   = path.parent_path() / "edges.bin";
        serialize_buffer_ver1(buffer_name, buffer, graph->getEdgesBuffer()->size());
        graph_json[EDGES_KEY] = "edges.bin";
        graph->getEdgesBuffer()->unmapHostPointers();
    }

    std::ofstream o(path);
    o << std::setw(4) << graph_json << std::endl;
}

ATCG_INLINE void serialize_script_ver1(const atcg::ref_ptr<Script>& script, const std::filesystem::path& path)
{
    auto path_  = path;
    auto source = script->getSource();
    std::ofstream stream(path_.replace_extension(".py"));
    stream << source;
    stream.close();
}

ATCG_INLINE void serialize_scene_ver1(const atcg::ref_ptr<Scene>& scene, const std::filesystem::path& path)
{
    // TODO: Templates ?
    auto path_ = path;
    atcg::Serialization::SceneSerializer serializer(scene);
    serializer.serialize(path_.replace_extension(".scene").string());
}

ATCG_INLINE void serialize_shader_ver1(const atcg::ref_ptr<Shader>& shader, const std::filesystem::path& path)
{
    nlohmann::json shader_json;

    shader_json["Version"] = "1.0";

    auto path_ = path;

    if(shader->isComputeShader())
    {
        path_ = path_.replace_extension(".glsl");
        std::ofstream stream(path_);
        stream << shader->getSource(ShaderType::COMPUTE);
        stream.close();
        shader_json["Compute"] = path_.filename();
    }
    else
    {
        if(shader->hasGeometryShader())
        {
            path_ = path_.replace_extension(".gs");
            std::ofstream stream(path_);
            stream << shader->getSource(ShaderType::GEOMETRY);
            stream.close();

            shader_json["Geometry"] = path_.filename();
        }

        {
            path_ = path_.replace_extension(".vs");
            std::ofstream stream(path_);
            stream << shader->getSource(ShaderType::VERTEX);
            stream.close();
            shader_json["Vertex"] = path_.filename();
        }

        {
            path_ = path_.replace_extension(".fs");
            std::ofstream stream(path_);
            stream << shader->getSource(ShaderType::FRAGMENT);
            stream.close();
            shader_json["Fragment"] = path_.filename();
        }
    }

    std::ofstream o(path_.replace_extension(".json"));
    o << std::setw(4) << shader_json << std::endl;
}

ATCG_INLINE void serialize_texture3d_ver1(const atcg::ref_ptr<Texture3D>& texture, const std::filesystem::path& path)
{
    auto path_ = path;

    torch::Tensor texture_data = texture->getData(atcg::CPU);

    nlohmann::json texture_json;

    texture_json["Version"] = "1.0";
    path_                   = path_.replace_extension(".bin");
    texture_json["Path"]    = path_.filename();
    texture_json["Width"]   = texture->width();
    texture_json["Height"]  = texture->height();
    texture_json["Depth"]   = texture->depth();
    texture_json["Format"]  = textureFormatToString(texture->getSpecification().format);
    texture_json["Wrap"]    = textureWrapModeToString(texture->getSpecification().sampler.wrap_mode);
    texture_json["Filter"]  = textureFilterModeToString(texture->getSpecification().sampler.filter_mode);
    texture_json["MipMap"]  = texture->getSpecification().sampler.mip_map;

    serialize_buffer_ver1(path_,
                          (const char*)texture_data.data_ptr(),
                          texture_data.numel() * texture_data.element_size());

    std::ofstream o(path_.replace_extension(".json"));
    o << std::setw(4) << texture_json << std::endl;
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
            serialize_material_ver1(std::dynamic_pointer_cast<atcg::Material>(asset),
                                    material_path / (data.name + ".mat"));
        }
        break;
        case AssetType::Texture2D:
        {
            auto texture_path = path / "textures" / std::to_string(asset->handle);
            std::filesystem::create_directories(texture_path);
            serialize_texture2d_ver1(std::dynamic_pointer_cast<atcg::Texture2D>(asset), texture_path / data.name);
        }
        break;
        case AssetType::Texture3D:
        {
            auto texture_path = path / "textures" / std::to_string(asset->handle);
            std::filesystem::create_directories(texture_path);
            serialize_texture3d_ver1(std::dynamic_pointer_cast<atcg::Texture3D>(asset), texture_path / data.name);
        }
        break;
        case AssetType::Scene:
        {
            auto scene_path = path / "scenes" / std::to_string(asset->handle);
            std::filesystem::create_directories(scene_path);
            serialize_scene_ver1(std::dynamic_pointer_cast<Scene>(asset), scene_path / data.name);
        }
        break;
        case AssetType::Script:
        {
            auto script_path = path / "scripts" / std::to_string(asset->handle);
            std::filesystem::create_directories(script_path);
            serialize_script_ver1(std::dynamic_pointer_cast<Script>(asset), script_path / data.name);
        }
        break;
        case AssetType::Shader:
        {
            auto shader_path = path / "shader" / std::to_string(asset->handle);
            std::filesystem::create_directories(shader_path);
            serialize_shader_ver1(std::dynamic_pointer_cast<Shader>(asset), shader_path / data.name);
        }
        break;
    }
}
}    // namespace detail

void AssetExporter::exportAsset(const std::filesystem::path& path,
                                const atcg::ref_ptr<Asset>& asset,
                                const AssetMetaData& data)
{
    if(asset && data.serialize) detail::export_asset_ver1(path, asset, data);
}
}    // namespace atcg