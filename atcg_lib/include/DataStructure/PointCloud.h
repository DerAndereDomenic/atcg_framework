#pragma once

#include <vector>
#include <OpenMesh/OpenMesh.h>
#include <OpenMesh/Core/Mesh/Traits.hh>
#include <Renderer/VertexArray.h>
#include <Math/Utils.h>

#include <OpenMesh/GLMTraits.h>

namespace atcg
{
template<class Traits = GLMTraits>
class PointCloudT
{
public:
    typedef typename Traits::Point Point;
    typedef typename Traits::Normal Normal;
    typedef float Scalar;
    typedef typename Traits::Color Color;
    typedef OpenMesh::VertexHandle VertexHandle;

    /**
     * @brief Create a pointcloud
     */
    PointCloudT() = default;

    /**
     * @brief Destroy the pointcloud object
     */
    ~PointCloudT() = default;

    /**
     * @brief Add a vertex
     *
     * @param p The point
     *
     * @returns The handle to this vertex
     */
    VertexHandle add_vertex(const Point& p);

    /**
     * @brief Set the point of a vertex
     *
     * @param vh The VertexHandle
     * @param p The point
     */
    void set_point(const VertexHandle& handle, const Point& p);

    /**
     * @brief Set the normal
     *
     * @param vh The VertexHandle
     * @param normal The normal
     */
    void set_normal(const VertexHandle& vh, const Normal& normal);

    /**
     * @brief Set the color
     *
     * @param vh The VertexHandle
     * @param color The color
     */
    void set_color(const VertexHandle& vh, const Color& color);

    /**
     * @brief Set the color of all points
     * @param color The color
     */
    void setColor(const Color& color);

    /**
     * @brief Get the point of a vertex
     *
     * @param vh The VertexHandle
     * @returns The point
     */
    Point point(const VertexHandle& vh);

    /**
     * @brief Get the normal of a vertex
     *
     * @param vh The VertexHandle
     * @returns The normal
     */
    Normal normal(const VertexHandle& vh);

    /**
     * @brief Get the color of a vertex
     *
     * @param vh The VertexHandle
     * @return The color
     */
    Color color(const VertexHandle& vh);

    /**
     * @brief Uploads the data onto the gpu
     */
    void uploadData();

    /**
     * @brief Get the point cloud as Nx3 row matrix
     *
     * @return The data points as matrix
     */
    RowMatrix asMatrix();

    /**
     * @brief Set the point data from a matrix
     *
     * @param points The point matrix
     */
    void fromMatrix(const RowMatrix& points);

    /**
     * @brief Get the Vertex Array object
     *
     * @return atcg::ref_ptr<VertexArray> The vao
     */
    inline atcg::ref_ptr<VertexArray> getVertexArray() const { return _vao; }

    /**
     * @brief Apply a model matrix to all points
     *
     * @param transform The transform matrix
     */
    void applyTransform(const glm::mat4& transform);

    /**
     * @brief Get the number of vertices
     *
     * @returns The number of vertices
     */
    inline size_t n_vertices() const { return _vertices.size(); }

    // Iterators
    std::vector<VertexHandle>::iterator vertices_begin() { return _vertices.begin(); }
    std::vector<VertexHandle>::iterator vertices_end() { return _vertices.end(); }

    std::vector<VertexHandle>::const_iterator vertices_begin() const { return _vertices.begin(); }
    std::vector<VertexHandle>::const_iterator vertices_end() const { return _vertices.end(); }

private:
    std::vector<VertexHandle> _vertices;

    std::vector<Point> _points;
    std::vector<Normal> _normals;
    std::vector<Color> _colors;

    atcg::ref_ptr<VertexArray> _vao;
};

///
/// Implementation
///

template<class Traits>
typename PointCloudT<Traits>::VertexHandle PointCloudT<Traits>::add_vertex(const PointCloudT<Traits>::Point& p)
{
    typename PointCloudT<Traits>::VertexHandle vh(static_cast<int>(_vertices.size()));
    _vertices.push_back(vh);
    _normals.push_back(typename PointCloudT<Traits>::Normal {1, 0, 0});
    _colors.push_back(typename PointCloudT<Traits>::Color {255, 255, 255});
    _points.push_back(p);

    return vh;
}

template<class Traits>
void PointCloudT<Traits>::set_point(const PointCloudT<Traits>::VertexHandle& vh, const PointCloudT<Traits>::Point& p)
{
    _points[vh.idx()] = p;
}

template<class Traits>
void PointCloudT<Traits>::set_normal(const PointCloudT<Traits>::VertexHandle& vh,
                                     const PointCloudT<Traits>::Normal& normal)
{
    _normals[vh.idx()] = normal;
}

template<class Traits>
void PointCloudT<Traits>::set_color(const PointCloudT<Traits>::VertexHandle& vh,
                                    const PointCloudT<Traits>::Color& color)
{
    _colors[vh.idx()] = color;
}

template<class Traits>
void PointCloudT<Traits>::setColor(const PointCloudT<Traits>::Color& color)
{
    for(uint32_t i = 0; i < _colors.size(); ++i) _colors[i] = color;
}

template<class Traits>
typename PointCloudT<Traits>::Point PointCloudT<Traits>::point(const PointCloudT<Traits>::VertexHandle& vh)
{
    return _points[vh.idx()];
}

template<class Traits>
typename PointCloudT<Traits>::Normal PointCloudT<Traits>::normal(const PointCloudT<Traits>::VertexHandle& vh)
{
    return _normals[vh.idx()];
}

template<class Traits>
typename PointCloudT<Traits>::Color PointCloudT<Traits>::color(const PointCloudT<Traits>::VertexHandle& vh)
{
    return _colors[vh.idx()];
}

template<class Traits>
void PointCloudT<Traits>::uploadData()
{
    std::vector<float> vertex_data;
    vertex_data.resize(_vertices.size() * 9);

    for(auto vertex: _vertices)
    {
        int32_t vertex_id              = vertex.idx();
        glm::vec3 pos                  = point(vertex);
        glm::vec3 norm                 = normal(vertex);
        glm::vec3 col                  = color(vertex);
        vertex_data[9 * vertex_id + 0] = pos[0];
        vertex_data[9 * vertex_id + 1] = pos[1];
        vertex_data[9 * vertex_id + 2] = pos[2];
        vertex_data[9 * vertex_id + 3] = norm[0];
        vertex_data[9 * vertex_id + 4] = norm[1];
        vertex_data[9 * vertex_id + 5] = norm[2];
        vertex_data[9 * vertex_id + 6] = static_cast<float>(col[0]) / 255.0f;
        vertex_data[9 * vertex_id + 7] = static_cast<float>(col[1]) / 255.0f;
        vertex_data[9 * vertex_id + 8] = static_cast<float>(col[2]) / 255.0f;
    }

    _vao = atcg::make_ref<VertexArray>();
    atcg::ref_ptr<VertexBuffer> vbo =
        atcg::make_ref<VertexBuffer>(vertex_data.data(), static_cast<uint32_t>(vertex_data.size() * sizeof(float)));
    vbo->setLayout({{ShaderDataType::Float3, "aPosition"},
                    {ShaderDataType::Float3, "aNormal"},
                    {ShaderDataType::Float3, "aColor"}});

    _vao->pushVertexBuffer(vbo);
}

template<class Traits>
RowMatrix PointCloudT<Traits>::asMatrix()
{
    RowMatrix S(n_vertices(), 3);
    uint32_t i = 0;
    for(auto vertex: _vertices)
    {
        glm::vec3 pos = point(vertex);
        S(i, 0)       = static_cast<double>(pos[0]);
        S(i, 1)       = static_cast<double>(pos[1]);
        S(i, 2)       = static_cast<double>(pos[2]);
        ++i;
    }

    return S;
}

template<class Traits>
void PointCloudT<Traits>::fromMatrix(const RowMatrix& points)
{
    _vertices.clear();
    _points.clear();
    _normals.clear();
    _colors.clear();

    for(uint32_t i = 0; i < points.rows(); ++i)
    {
        glm::vec3 pos {points(i, 0), points(i, 1), points(i, 2)};
        add_vertex(pos);
    }
}

template<class Traits>
void PointCloudT<Traits>::applyTransform(const glm::mat4& transform)
{
    glm::mat4 model_matrix = glm::inverse(glm::transpose(transform));
    for(auto vt = vertices_begin(); vt != vertices_end(); ++vt)
    {
        set_point(*vt, glm::vec3(transform * glm::vec4(point(*vt), 1.0f)));
        set_normal(*vt, glm::normalize(glm::vec3(model_matrix * glm::vec4(normal(*vt), 1.0f))));
    }
}

using PointCloud = PointCloudT<>;

namespace IO
{
/**
 * @brief Load a pointcloud
 *
 * @param path The path
 * @returns The pointcloud
 */
atcg::ref_ptr<PointCloud> read_pointcloud(const char* path);
}    // namespace IO
}    // namespace atcg