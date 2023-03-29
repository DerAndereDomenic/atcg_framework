#include <DataStructure/Mesh.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace atcg
{
void Mesh::uploadData()
{
    request_vertex_normals();
    request_face_normals();
    update_normals();

    std::vector<float> vertex_data;
    vertex_data.resize(n_vertices() * 9);

    std::vector<uint32_t> indices_data;
    indices_data.resize(n_faces() * 3);

    bool has_color = has_vertex_colors();

    for(auto vertex = vertices_begin(); vertex != vertices_end(); ++vertex)
    {
        int32_t vertex_id              = vertex->idx();
        glm::vec3 pos                  = point(*vertex);
        glm::vec3 normal               = calc_vertex_normal(*vertex);
        glm::vec3 col                  = has_color ? color(*vertex) : glm::vec3(1);
        vertex_data[9 * vertex_id + 0] = pos[0];
        vertex_data[9 * vertex_id + 1] = pos[1];
        vertex_data[9 * vertex_id + 2] = pos[2];
        vertex_data[9 * vertex_id + 3] = normal[0];
        vertex_data[9 * vertex_id + 4] = normal[1];
        vertex_data[9 * vertex_id + 5] = normal[2];
        vertex_data[9 * vertex_id + 6] = has_color ? static_cast<float>(col[0]) / 255.0f : 1.0f;
        vertex_data[9 * vertex_id + 7] = has_color ? static_cast<float>(col[1]) / 255.0f : 1.0f;
        vertex_data[9 * vertex_id + 8] = has_color ? static_cast<float>(col[2]) / 255.0f : 1.0f;
    }

    int32_t face_id = 0;
    for(auto face = faces_begin(); face != faces_end(); ++face)
    {
        int32_t vertex_id = 0;
        for(auto vertex = face->vertices().begin(); vertex != face->vertices().end(); ++vertex)
        {
            indices_data[3 * face_id + vertex_id] = vertex->idx();
            ++vertex_id;
        }
        ++face_id;
    }

    _vao = std::make_shared<atcg::VertexArray>();
    std::shared_ptr<atcg::VertexBuffer> vbo =
        std::make_shared<atcg::VertexBuffer>(vertex_data.data(),
                                             static_cast<uint32_t>(sizeof(float) * vertex_data.size()));
    vbo->setLayout({{atcg::ShaderDataType::Float3, "aPosition"},
                    {atcg::ShaderDataType::Float3, "aNormal"},
                    {atcg::ShaderDataType::Float3, "aColor"}});
    _vao->addVertexBuffer(vbo);

    std::shared_ptr<atcg::IndexBuffer> ibo =
        std::make_shared<atcg::IndexBuffer>(indices_data.data(), static_cast<uint32_t>(indices_data.size()));
    _vao->setIndexBuffer(ibo);
}

void Mesh::addBuffer(const std::shared_ptr<VertexBuffer>& buffer)
{
    _vao->addVertexBuffer(buffer);
}

void Mesh::calculateModelMatrix()
{
    glm::mat4 scale     = glm::scale(_scale);
    glm::mat4 translate = glm::translate(_position);
    glm::mat4 rotation  = glm::rotate(_rotation_angle, _rotation_axis);

    _model = translate * rotation * scale;
}

float Mesh::area(const Mesh::FaceHandle& f_handle) const
{
    return calc_face_area(f_handle);
}

float Mesh::area(const Mesh::VertexHandle& v_handle) const
{
    float total_area = 0.0f;
    VertexIter v_itter(*this, v_handle);
    for(auto f_it = v_itter->faces().begin(); f_it != v_itter->faces().end(); ++f_it) { total_area += area(*f_it); }
    return total_area;
}

float Mesh::total_area() const
{
    float total_area = 0.0f;

    for(auto f_it = faces_begin(); f_it != faces_end(); ++f_it) { total_area += area(*f_it); }

    return total_area;
}

void Mesh::setColor(const Color& color)
{
    for(auto v_it = vertices_begin(); v_it != vertices_end(); ++v_it) { set_color(*v_it, color); }
}

void Mesh::setColors(const Eigen::MatrixX3d& colors)
{
    for(auto v_it = vertices_begin(); v_it != vertices_end(); ++v_it)
    {
        Eigen::Vector3d c = colors.row(v_it->idx());
        Color color       = glm::vec3(c.x(), c.y(), c.z());
        set_color(*v_it, color);
    }
}

Mesh::HalfedgeHandle Mesh::opposite_halfedge_handle(const Mesh::FaceHandle& fh, const Mesh::VertexHandle& vh)
{
    for(auto he_it = cfh_ccwbegin(fh); he_it != cfh_ccwend(fh); ++he_it)
    {
        if(from_vertex_handle(*he_it) != vh && to_vertex_handle(*he_it) != vh) { return *he_it; }
    }

    // This can not happen but compiler wants a return value for every code path.
    return Mesh::HalfedgeHandle();
}

namespace IO
{
std::shared_ptr<Mesh> read_mesh(const char* path)
{
    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
    OpenMesh::IO::read_mesh(*mesh.get(), path);
    return mesh;
}
}    // namespace IO

}    // namespace atcg