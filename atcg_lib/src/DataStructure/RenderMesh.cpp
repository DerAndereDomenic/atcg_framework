#include <DataStructure/RenderMesh.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace atcg
{
    void RenderMesh::uploadData(const std::shared_ptr<TriMesh>& mesh)
    {
        mesh->request_vertex_normals();
        mesh->request_face_normals();
        mesh->update_normals();

        std::vector<float> vertex_data;
        vertex_data.resize(mesh->n_vertices() * 9);

        std::vector<uint32_t> indices_data;
        indices_data.resize(mesh->n_faces() * 3);

        bool has_color = mesh->has_vertex_colors();

        for(auto vertex = mesh->vertices_begin(); vertex != mesh->vertices_end(); ++vertex)
        {
            int32_t vertex_id = vertex->idx();
            OpenMesh::Vec3f pos = mesh->point(*vertex);
            OpenMesh::Vec3f normal = mesh->calc_vertex_normal(*vertex);
            OpenMesh::Vec3uc color = has_color ? mesh->color(*vertex) : OpenMesh::Vec3uc();
            vertex_data[9 * vertex_id + 0] = pos[0];
            vertex_data[9 * vertex_id + 1] = pos[1];
            vertex_data[9 * vertex_id + 2] = pos[2];
            vertex_data[9 * vertex_id + 3] = normal[0];
            vertex_data[9 * vertex_id + 4] = normal[1];
            vertex_data[9 * vertex_id + 5] = normal[2];
            vertex_data[9 * vertex_id + 6] = has_color ? static_cast<float>(color[0])/255.0f : 1.0f;
            vertex_data[9 * vertex_id + 7] = has_color ? static_cast<float>(color[1])/255.0f : 1.0f;
            vertex_data[9 * vertex_id + 8] = has_color ? static_cast<float>(color[2])/255.0f : 1.0f;
        }

        int32_t face_id = 0;
        for(auto face = mesh->faces_begin(); face != mesh->faces_end(); ++face)
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
        std::shared_ptr<atcg::VertexBuffer> vbo = std::make_shared<atcg::VertexBuffer>(vertex_data.data(), static_cast<uint32_t>(sizeof(float) * vertex_data.size()));
        vbo->setLayout({
            {atcg::ShaderDataType::Float3, "aPosition"},
            {atcg::ShaderDataType::Float3, "aNormal"},
            {atcg::ShaderDataType::Float3, "aColor"}
        });
        _vao->addVertexBuffer(vbo);

        std::shared_ptr<atcg::IndexBuffer> ibo = std::make_shared<atcg::IndexBuffer>(indices_data.data(), static_cast<uint32_t>(indices_data.size()));
        _vao->setIndexBuffer(ibo);
    }

    void RenderMesh::addBuffer(const std::shared_ptr<VertexBuffer>& buffer)
    {
        _vao->addVertexBuffer(buffer);
    }

    void RenderMesh::calculateModelMatrix()
    {
        glm::mat4 scale = glm::scale(_scale);
        glm::mat4 translate = glm::translate(_position);
        glm::mat4 rotation = glm::rotate(_rotation_angle, _rotation_axis);

        _model = translate * rotation * scale;
    }
}