#pragma once

#include <vector>
#include <OpenMesh/OpenMesh.h>
#include <OpenMesh/Core/Mesh/Traits.hh>
#include <Renderer/VertexArray.h>

namespace atcg
{
    template<class Traits = OpenMesh::DefaultTraits>
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
         * @brief Get the Vertex Array object
         * 
         * @return std::shared_ptr<VertexArray> The vao
         */
        inline std::shared_ptr<VertexArray> getVertexArray() const {return _vao;}

        /**
         * @brief Get the number of vertices
         * 
         * @returns The number of vertices
         */
        inline uint32_t n_vertices() const { return _vertices.size(); }
        
        //Iterators
        std::vector<VertexHandle>::iterator vertices_begin() { return _vertices.begin(); }
		std::vector<VertexHandle>::iterator vertices_end() { return _vertices.end(); }

		std::vector<VertexHandle>::const_iterator vertices_begin() const { return _vertices.begin(); }
		std::vector<VertexHandle>::const_iterator vertices_end()	const { return _vertices.end(); }

    private:
        std::vector<VertexHandle> _vertices;

        std::vector<Point> _points;
        std::vector<Normal> _normals;
        std::vector<Color> _colors;
        
        std::shared_ptr<VertexArray> _vao;
    };

    ///
    /// Implementation
    ///

    template<class Traits>
    typename PointCloudT<Traits>::VertexHandle PointCloudT<Traits>::add_vertex(const PointCloudT<Traits>::Point& p)
    {
        typename PointCloudT<Traits>::VertexHandle vh(_vertices.size());
        _vertices.push_back(vh);
        _normals.push_back(typename PointCloudT<Traits>::Normal{1,0,0});
        _colors.push_back(typename PointCloudT<Traits>::Color{0,0,0});
        _points.push_back(p);

        return vh;
    }

    template<class Traits>
    void PointCloudT<Traits>::set_point(const PointCloudT<Traits>::VertexHandle& handle, const PointCloudT<Traits>::Point& p)
    {
        _points[vh.idx()] = p;
    }

    template<class Traits>
    void PointCloudT<Traits>::set_normal(const PointCloudT<Traits>::VertexHandle& vh, const PointCloudT<Traits>::Normal& normal)
    {
        _normals[vh.idx()] = normal;
    }

    template<class Traits>
    void PointCloudT<Traits>::set_color(const PointCloudT<Traits>::VertexHandle& vh, const PointCloudT<Traits>::Color& color)
    {
        _colors[vh.idx()] = color;
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
        
        for(auto vertex : _vertices)
        {
            int32_t vertex_id = vertex.idx();
            OpenMesh::Vec3f pos = point(vertex);
            OpenMesh::Vec3f norm = normal(vertex);
            OpenMesh::Vec3uc col = color(vertex);
            vertex_data[9 * vertex_id + 0] = pos[0];
            vertex_data[9 * vertex_id + 1] = pos[1];
            vertex_data[9 * vertex_id + 2] = pos[2];
            vertex_data[9 * vertex_id + 3] = norm[0];
            vertex_data[9 * vertex_id + 4] = norm[1];
            vertex_data[9 * vertex_id + 5] = norm[2];
            vertex_data[9 * vertex_id + 6] = static_cast<float>(col[0])/255.0f;
            vertex_data[9 * vertex_id + 7] = static_cast<float>(col[1])/255.0f;
            vertex_data[9 * vertex_id + 8] = static_cast<float>(col[2])/255.0f;
        }

        _vao = std::make_shared<VertexArray>();
        std::shared_ptr<VertexBuffer> vbo = std::make_shared<VertexBuffer>(vertex_data.data(), static_cast<uint32_t>(vertex_data.size() * sizeof(float)));
        vbo->setLayout({
            {ShaderDataType::Float3, "aPosition"},
            {ShaderDataType::Float3, "aNormal"},
            {ShaderDataType::Float3, "aColor"}
        });

        _vao->addVertexBuffer(vbo);
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
        std::shared_ptr<PointCloud> read_pointcloud(const char* path);
    }
}