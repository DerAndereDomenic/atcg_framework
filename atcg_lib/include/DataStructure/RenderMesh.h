#pragma once

#include <glm/glm.hpp>
#include <Renderer/VertexArray.h>
#include <OpenMesh/OpenMesh.h>

namespace atcg
{
    /**
     * @brief This class basically models a triangle soup.
     * This structure is only used to hold the GPU buffers for rendering the mesh
     * For the half-link data structure for advanced algorithms, use the TriMesh class from OpenMesh
     * When uploading the data from a TriMesh to this RenderMesh class, the attributes are layed out in the following structure:
     * aPosition - Float3
     * aNormal - Float3
     * aVertexColor - Float3
     * You can add custom properties via the addBuffer() function
     */
    class RenderMesh
    {
    public:
        /**
         * @brief Construct a new Render Mesh object
         */
        RenderMesh() = default;

        /**
         * @brief Destroy the Render Mesh object
         */
        ~RenderMesh() = default;

        /**
         * @brief Upload the data from a TriMesh onto the GPU for rendering
         * 
         * @param mesh The mesh to render
         */
        void uploadData(const std::shared_ptr<TriMesh>& mesh);

        /**
         * @brief Add a custom vertex buffer
         * 
         * @param buffer The buffer
         */
        void addBuffer(const std::shared_ptr<VertexBuffer>& buffer);

        /**
         * @brief Get the Position
         * 
         * @return glm::vec3 The position
         */
        inline glm::vec3 getPosition() const {return _position;}

        /**
         * @brief Get the Scale
         * 
         * @return glm::vec3 The scale
         */
        inline glm::vec3 getScale() const {return _scale;}

        /**
         * @brief Get the Model
         * 
         * @return glm::mat4 The model
         */
        inline glm::mat4 getModel() const {return _model;}

        /**
         * @brief Set the Position
         * 
         * @param position The position
         */
        inline void setPosition(const glm::vec3& position) {_position = position; calculateModelMatrix();}

        /**
         * @brief Set the Scale
         * 
         * @param scale The scale
         */
        inline void setScale(const glm::vec3& scale) {_scale = scale; calculateModelMatrix();}

        /**
         * @brief Set the Rotation
         * 
         * @param axis The rotation axis
         * @param angle The rotation angle
         */
        inline void setRotation(const glm::vec3& axis, const float& angle) {_rotation_axis = axis; _rotation_angle = angle; calculateModelMatrix();}

        /**
         * @brief Get the Vertex Array object
         * 
         * @return std::shared_ptr<VertexArray> The vao
         */
        inline std::shared_ptr<VertexArray> getVertexArray() const {return _vao;}

    private:
        void calculateModelMatrix();

        glm::vec3 _position = glm::vec3(0);
        glm::vec3 _scale = glm::vec3(1);
        glm::mat4 _model = glm::mat4(1);

        glm::vec3 _rotation_axis = glm::vec3(0,1,0);
        float _rotation_angle = 0;

        std::shared_ptr<VertexArray> _vao;
    };
}