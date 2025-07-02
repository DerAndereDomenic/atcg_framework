#include <DataStructure/GraphLoader.h>
#include <DataStructure/Graph.h>
#include <tiny_obj_loader.h>

#include <Scene/Scene.h>
#include <Scene/Entity.h>
#include <Scene/Components.h>

namespace atcg
{
namespace IO
{
namespace detail
{

tinyobj::ObjReader read_file(const std::string& path)
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "";    // Path to material files

    tinyobj::ObjReader reader;

    if(!reader.ParseFromFile(path, reader_config))
    {
        if(!reader.Error().empty())
        {
            ATCG_ERROR("TinyObjReader: {0}", reader.Error());
        }
        return reader;
    }

    if(!reader.Warning().empty())
    {
        ATCG_WARN("TinyObjReader: {0}", reader.Warning());
    }

    return reader;
}

template<typename Loader>
void load_vertices(const tinyobj::attrib_t& attrib, std::vector<atcg::Vertex>& out_vertices, Loader loader)
{
    struct less
    {
        bool operator()(const tinyobj::index_t& lhs, const tinyobj::index_t& rhs) const
        {
            if(lhs.vertex_index != rhs.vertex_index) return lhs.vertex_index < rhs.vertex_index;
            if(lhs.normal_index != rhs.normal_index) return lhs.normal_index < rhs.normal_index;
            return lhs.texcoord_index < rhs.texcoord_index;
        }
    };
    std::map<tinyobj::index_t, uint32_t, less> index_map;

    auto map_index = [&](const tinyobj::index_t& index)
    {
        auto it = index_map.find(index);
        // Vertex not yet added
        if(it == index_map.end())
        {
            it = index_map.insert({index, static_cast<uint32_t>(index_map.size())}).first;

            atcg::Vertex vertex;
            if(!attrib.vertices.empty() && index.vertex_index != -1)
            {
                vertex.position = glm::vec3(attrib.vertices[3 * index.vertex_index + 0],
                                            attrib.vertices[3 * index.vertex_index + 1],
                                            attrib.vertices[3 * index.vertex_index + 2]);
            }

            if(!attrib.normals.empty() && index.normal_index != -1)
            {
                vertex.normal = glm::vec3(attrib.normals[3 * index.normal_index + 0],
                                          attrib.normals[3 * index.normal_index + 1],
                                          attrib.normals[3 * index.normal_index + 2]);
            }

            if(!attrib.texcoords.empty() && index.texcoord_index != -1)
            {
                vertex.uv = glm::vec3(attrib.texcoords[2 * index.texcoord_index + 0],
                                      attrib.texcoords[2 * index.texcoord_index + 1],
                                      0.0f);
            }

            if(!attrib.texcoord_ws.empty() && index.texcoord_index != -1)
            {
                vertex.uv.z = attrib.texcoord_ws[index.texcoord_index];
            }

            if(!attrib.colors.empty() && index.vertex_index != -1)
            {
                vertex.color = glm::vec3(attrib.colors[3 * index.vertex_index + 0],
                                         attrib.colors[3 * index.vertex_index + 1],
                                         attrib.colors[3 * index.vertex_index + 2]);
            }

            out_vertices.push_back(vertex);
        }
        return it->second;
    };

    loader(map_index);
}

void load_mesh_data(const tinyobj::attrib_t& attrib,
                    const tinyobj::shape_t& shape,
                    std::vector<atcg::Vertex>& out_vertices,
                    std::vector<glm::u32vec3>& out_indices)
{
    // Iterate over every face
    load_vertices(attrib,
                  out_vertices,
                  [&](const std::function<uint32_t(const tinyobj::index_t& index)> map_index)
                  {
                      size_t index_offset = 0;
                      for(size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
                      {
                          size_t fv = size_t(shape.mesh.num_face_vertices[f]);

                          if(fv != 3)
                          {
                              ATCG_WARN("Detected a face with {0} vertices", fv);
                          }

                          glm::u32vec3 index;
                          for(size_t v = 0; v < fv; ++v)
                          {
                              tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                              // Check if idx is already used and find appropriate index
                              size_t vertex_idx = map_index(idx);
                              index[v]          = vertex_idx;
                          }

                          out_indices.push_back(index);

                          // Calculate tangent vector
                          glm::vec3 pos1 = out_vertices[index[0]].position;
                          glm::vec3 pos2 = out_vertices[index[1]].position;
                          glm::vec3 pos3 = out_vertices[index[2]].position;

                          glm::vec2 uv1 = out_vertices[index[0]].uv;
                          glm::vec2 uv2 = out_vertices[index[1]].uv;
                          glm::vec2 uv3 = out_vertices[index[2]].uv;

                          glm::vec3 edge1 = pos2 - pos1;
                          glm::vec3 edge2 = pos3 - pos1;

                          glm::vec2 deltaUV1 = uv2 - uv1;
                          glm::vec2 deltaUV2 = uv3 - uv1;

                          float det = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

                          if(std::abs(det) >= 1e-5f)
                          {
                              float invdet = 1.0f / det;
                              glm::vec3 tangent;
                              tangent.x = det * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
                              tangent.y = det * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
                              tangent.z = det * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
                              tangent   = glm::normalize(tangent);

                              out_vertices[index[0]].tangent += tangent;
                              out_vertices[index[1]].tangent += tangent;
                              out_vertices[index[2]].tangent += tangent;
                          }

                          index_offset += fv;
                      }

                      for(auto& vtx: out_vertices)
                      {
                          float len = glm::length2(vtx.tangent);
                          if(len >= 1e-5f)
                          {
                              vtx.tangent = vtx.tangent / glm::sqrt(len);
                          }
                      }
                  });
}

void load_pointcloud_data(const tinyobj::attrib_t& attrib,
                          const tinyobj::shape_t& shape,
                          std::vector<atcg::Vertex>& out_vertices)
{
    // Iterate over every face
    load_vertices(attrib,
                  out_vertices,
                  [&](const std::function<uint32_t(const tinyobj::index_t& index)> map_index)
                  {
                      // Iterate over every vertex
                      for(size_t v = 0; v < shape.points.indices.size(); ++v)
                      {
                          // Check if idx is already used and find appropriate index
                          size_t vertex_idx = map_index(shape.points.indices[v]);
                      }
                  });
}

void load_line_data(const tinyobj::attrib_t& attrib,
                    const tinyobj::shape_t& shape,
                    std::vector<atcg::Vertex>& out_vertices,
                    std::vector<atcg::Edge>& out_edges)
{
    // Iterate over every face
    load_vertices(attrib,
                  out_vertices,
                  [&](const std::function<uint32_t(const tinyobj::index_t& index)>& map_index)
                  {
                      // Iterate over every vertex
                      size_t index_offset = 0;
                      for(size_t l = 0; l < shape.lines.num_line_vertices.size(); ++l)
                      {
                          // Check if idx is already used and find appropriate index
                          size_t lv = size_t(shape.lines.num_line_vertices[l]);
                          if(lv != 2)
                          {
                              ATCG_WARN("Detected a line with {0} vertices", lv);
                          }

                          glm::vec2 index;
                          for(size_t v = 0; v < lv; ++v)
                          {
                              tinyobj::index_t idx = shape.lines.indices[index_offset + v];

                              // Check if idx is already used and find appropriate index
                              size_t vertex_idx = map_index(idx);
                              index[v]          = vertex_idx;
                          }

                          atcg::Edge edge {index, glm::vec3(1), 1.0f};
                          out_edges.push_back(edge);

                          index_offset += lv;
                      }
                  });
}

}    // namespace detail
}    // namespace IO

atcg::ref_ptr<Graph> IO::read_any(const std::string& path)
{
    tinyobj::ObjReader reader = detail::read_file(path);

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    // Special case -> no face information is given. Techniqually this is not a valid obj but in most cases it is
    // assumed to be pointcloud data -> load all vertices. No auxilarry information is loaded because it is missing.
    if(shapes.size() == 0)
    {
        std::vector<Vertex> vertices;
        uint32_t num_vertices = attrib.vertices.size() / 3;
        std::cout << num_vertices << "\n";

        for(int i = 0; i < num_vertices; ++i)
        {
            vertices.push_back(atcg::Vertex(
                glm::vec3(attrib.vertices[3 * i + 0], attrib.vertices[3 * i + 1], attrib.vertices[3 * i + 2])));
        }

        return atcg::Graph::createPointCloud(vertices);
    }

    auto shape = shapes[0];    // We assume that these functions are only called with one object per obj. If not, we
                               // choose the first one as representitive for the whole collection.

    if(shape.mesh.indices.size() > 0)
    {
        return read_mesh(path);
    }

    if(shape.lines.indices.size() > 0)
    {
        return read_lines(path);
    }

    if(shape.points.indices.size() > 0)
    {
        return read_pointcloud(path);
    }

    return nullptr;
}

atcg::ref_ptr<Graph> IO::read_mesh(const std::string& path)
{
    tinyobj::ObjReader reader = detail::read_file(path);

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    std::vector<atcg::Vertex> vertices;
    std::vector<glm::u32vec3> faces;

    for(const auto& shape: shapes)
    {
        detail::load_mesh_data(attrib, shape, vertices, faces);
    }

    return atcg::Graph::createTriangleMesh(vertices, faces);
}

atcg::ref_ptr<Graph> IO::read_pointcloud(const std::string& path)
{
    tinyobj::ObjReader reader = detail::read_file(path);

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    std::vector<atcg::Vertex> vertices;

    for(const auto& shape: shapes)
    {
        detail::load_pointcloud_data(attrib, shape, vertices);
    }

    return atcg::Graph::createPointCloud(vertices);
}

atcg::ref_ptr<Graph> IO::read_lines(const std::string& path)
{
    tinyobj::ObjReader reader = detail::read_file(path);

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    std::vector<atcg::Vertex> vertices;
    std::vector<atcg::Edge> edges;

    for(const auto& shape: shapes)
    {
        detail::load_line_data(attrib, shape, vertices, edges);
    }

    return atcg::Graph::createGraph(vertices, edges);
}

atcg::ref_ptr<Scene> IO::read_scene(const std::string& path)
{
    atcg::ref_ptr<Scene> scene = atcg::make_ref<Scene>();

    tinyobj::ObjReader reader = detail::read_file(path);

    const auto& attrib        = reader.GetAttrib();
    const auto& shapes        = reader.GetShapes();
    const auto& materials_obj = reader.GetMaterials();

    std::vector<AssetHandle> materials(materials_obj.size());

    for(uint32_t i = 0; i < materials_obj.size(); ++i)
    {
        const tinyobj::material_t& mat_obj = materials_obj[i];
        auto material                      = atcg::make_ref<atcg::Material>();

        if(mat_obj.diffuse_texname != "")
        {
            float gamma  = mat_obj.diffuse_texopt.colorspace == "linear" ? 1.0f : 2.2f;
            auto img     = IO::imread(mat_obj.diffuse_texname, gamma);
            auto texture = atcg::Texture2D::create(img);
            material->setDiffuseTexture(texture);
        }
        else
        {
            material->setDiffuseColor(glm::make_vec3(mat_obj.diffuse));
        }

        if(mat_obj.metallic_texname != "")
        {
            auto img     = IO::imread(mat_obj.metallic_texname);
            auto texture = atcg::Texture2D::create(img);
            material->setMetallicTexture(texture);
        }
        else
        {
            material->setMetallic(mat_obj.metallic);
        }

        if(mat_obj.roughness_texname != "")
        {
            auto img     = IO::imread(mat_obj.roughness_texname);
            auto texture = atcg::Texture2D::create(img);
            material->setRoughnessTexture(texture);
        }
        else
        {
            material->setRoughness(mat_obj.roughness);
        }

        if(mat_obj.bump_texname != "")
        {
            auto img     = IO::imread(mat_obj.bump_texname);
            auto texture = atcg::Texture2D::create(img);
            material->setNormalTexture(texture);
        }

        AssetMetaData data;
        data.type      = AssetType::Material;
        data.name      = mat_obj.name;
        data.file_path = data.name + ".mat";
        materials[i]   = AssetManager::registerAsset(material, data);
    }

    for(const auto& shape: shapes)
    {
        if(shape.lines.num_line_vertices.size() != 0)
        {
            // Load a line collection
            std::vector<atcg::Vertex> vertices;
            std::vector<atcg::Edge> edges;
            detail::load_line_data(attrib, shape, vertices, edges);

            auto entity = scene->createEntity(shape.name);
            entity.addComponent<atcg::TransformComponent>();
            entity.addComponent<atcg::GeometryComponent>(atcg::Graph::createGraph(vertices, edges));
            entity.addComponent<atcg::EdgeCylinderRenderComponent>();
        }
        else if(shape.points.indices.size() != 0)
        {
            // Load a point cloud
            std::vector<atcg::Vertex> vertices;
            detail::load_pointcloud_data(attrib, shape, vertices);

            auto entity = scene->createEntity(shape.name);
            entity.addComponent<atcg::TransformComponent>();
            entity.addComponent<atcg::GeometryComponent>(atcg::Graph::createPointCloud(vertices));
            entity.addComponent<atcg::PointSphereRenderComponent>();
        }
        else
        {
            // Load a triangle mesh
            std::vector<atcg::Vertex> vertices;
            std::vector<glm::u32vec3> faces;
            detail::load_mesh_data(attrib, shape, vertices, faces);

            auto entity = scene->createEntity(shape.name);
            entity.addComponent<atcg::TransformComponent>();
            entity.addComponent<atcg::GeometryComponent>(atcg::Graph::createTriangleMesh(vertices, faces));
            auto& renderer = entity.addComponent<atcg::MeshRenderComponent>();
            if(shape.mesh.material_ids[0] != -1)
            {
                renderer.material_handle = materials[shape.mesh.material_ids[0]];
            }
        }
    }

    return scene;
}
}    // namespace atcg