#pragma once

#include <Core/glm.h>
#include <Renderer/VertexArray.h>
#include <OpenMesh/OpenMesh.h>
#include <Eigen/Dense>

namespace atcg
{

struct EulerAngle
{
    float rx, ry, rz;
};

/**
 * @brief This class basically models a triangle soup.
 * This structure is only used to hold the GPU buffers for rendering the mesh
 * For the half-link data structure for advanced algorithms, use the TriMesh class from OpenMesh
 * When uploading the data from a TriMesh to this RenderMesh class, the attributes are layed out in the following
 * structure: aPosition - Float3 aNormal - Float3 aVertexColor - Float3 You can add custom properties via the
 * addBuffer() function
 */
class Mesh : public TriMesh
{
public:
    /**
     * @brief Construct a new Render Mesh object
     */
    Mesh() = default;

    /**
     * @brief Destroy the Render Mesh object
     */
    ~Mesh() = default;

    /**
     * @brief Upload the data from a TriMesh onto the GPU for rendering
     */
    void uploadData();

    /**
     * @brief Add a custom vertex buffer
     *
     * @param buffer The buffer
     */
    void addBuffer(const atcg::ref_ptr<VertexBuffer>& buffer);

    /**
     * @brief Calcualte the area of a triangle
     *
     * @return float The area
     */
    float area(const Mesh::FaceHandle& f_handle) const;

    /**
     * @brief Get the area of the triangles adjacent to the vertex
     *
     * @param v_handle The vertex handle
     * @return float The area
     */
    float area(const Mesh::VertexHandle& v_handle) const;

    /**
     * @brief Get the total area of the mesh
     *
     * @return float The total area
     */
    float total_area() const;

    /**
     * @brief Set the color of all vertices
     *
     * @param color The color
     */
    void setColor(const Color& color);

    /**
     * @brief Set the color of all vertices
     *
     * @param colors A matrix with vertex colors
     */
    void setColors(const Eigen::MatrixX3d& colors);

    /**
     * @brief Get the handle to the opposite vertex on the face
     *
     * @param fh The face handle
     * @param vh The vertex handle
     *
     * @return The halfedge handle
     */
    Mesh::HalfedgeHandle opposite_halfedge_handle(const Mesh::FaceHandle& fh, const Mesh::VertexHandle& vh);

    /**
     * @brief Get the Position
     *
     * @return glm::vec3 The position
     */
    inline glm::vec3 getPosition() const { return _position; }

    /**
     * @brief Get the Scale
     *
     * @return glm::vec3 The scale
     */
    inline glm::vec3 getScale() const { return _scale; }

    /**
     * @brief Get the Model
     *
     * @return glm::mat4 The model
     */
    inline glm::mat4 getModel() const { return _model; }

    /**
     * @brief Set the model matrix
     *
     * @param model The new model matrix
     */
    inline void setModel(const glm::mat4 model)
    {
        _model = model;
        decomposeModelMatrix();
    }

    /**
     * @brief Set the Position
     *
     * @param position The position
     */
    inline void setPosition(const glm::vec3& position)
    {
        _position = position;
        calculateModelMatrix();
    }

    /**
     * @brief Set the Scale
     *
     * @param scale The scale
     */
    inline void setScale(const glm::vec3& scale)
    {
        _scale = scale;
        calculateModelMatrix();
    }

    /**
     * @brief Set the Rotation
     *
     * @param rotation The rotation described by three euler angles
     */
    inline void setRotation(const EulerAngle& rotation)
    {
        _rotation = rotation;
        calculateModelMatrix();
    }

    /**
     * @brief Get the Vertex Array object
     *
     * @return atcg::ref_ptr<VertexArray> The vao
     */
    inline atcg::ref_ptr<VertexArray> getVertexArray() const { return _vao; }

private:
    void calculateModelMatrix();
    void decomposeModelMatrix();

    glm::vec3 _position = glm::vec3(0);
    glm::vec3 _scale    = glm::vec3(1);
    glm::mat4 _model    = glm::mat4(1);

    EulerAngle _rotation;

    atcg::ref_ptr<VertexArray> _vao;
};

using VertexHandle = Mesh::VertexHandle;
using FaceHandle   = Mesh::FaceHandle;
using EdgeHandle   = Mesh::EdgeHandle;

namespace IO
{
/**
 * @brief Load a mesh
 *
 * @param path The path
 * @param colors If vertex colors should be loaded
 * @returns The mesh
 */
atcg::ref_ptr<Mesh> read_mesh(const char* path, bool colors = false);
}    // namespace IO
}    // namespace atcg