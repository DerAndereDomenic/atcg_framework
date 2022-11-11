#pragma once

#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Shader.h>
#include <Renderer/PerspectiveCamera.h>
#include <DataStructure/Mesh.h>
#include <DataStructure/Grid.h>
#include <DataStructure/PointCloud.h>

#include <memory>

#include <glm/glm.hpp>

namespace atcg
{
    /**
     * @brief This class models a renderer
     */
    class Renderer
    {
    public:
        /**
         * @brief Initializes the renderer (should not be called by the client!)
         * 
         * @param width The width
         * @param height The height
         */
        static void init(uint32_t width, uint32_t height);

        /**
         * @brief Finished the currently drawn frame (should not be called by client!) 
         */
        static void finishFrame();

        /**
         * @brief Set the clear color
         * 
         * @param color The clear color
         */
        static void setClearColor(const glm::vec4& color);

        /**
         * @brief Set the size of rendered points
         * 
         * @param size The size 
         */
        static void setPointSize(const float& size);

        /**
         * @brief Change the viewport of the renderer
         * 
         * @param x The viewport x location
         * @param y The viewport y location
         * @param width The width
         * @param height The height
         */
        static void setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height);

        /**
         * @brief Use the default screen fbo 
         */
        static void useScreenBuffer();

        /**
         * @brief Render a vao
         * NEEDS to have an index buffer
         * 
         * @param vao The vertex array
         * @param shader The shader
         * @param camera The camera
         */
        static void draw(const std::shared_ptr<VertexArray>& vao,
                                const std::shared_ptr<Shader>& shader,
                                const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Render a mesh
         * 
         * @param mesh The mesh
         * @param shader The shader
         * @param camera The camera
         */
        static void draw(const std::shared_ptr<Mesh>& mesh,
                                const std::shared_ptr<Shader>& shader,
                                const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Draw a pointcloud
         * 
         * @param cloud The pointcloud
         * @param shader The shader
         * @param camera The camera
         */
        static void draw(const std::shared_ptr<PointCloud>& cloud, 
                         const std::shared_ptr<Shader>& shader, 
                         const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Render a vao as points
         * NEEDS to have an index buffer
         * 
         * @param vao The vertex array
         * @param color The color
         * @param shader The shader
         * @param camera The camera
         */
        static void drawPoints(const std::shared_ptr<VertexArray>& vao,
                                      const glm::vec3& color,
                                      const std::shared_ptr<Shader>& shader,
                                      const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Render a mesh as points
         * 
         * @param mesh The mesh
         * @param color The color
         * @param shader The shader
         * @param camera The camera
         */
        static void drawPoints(const std::shared_ptr<Mesh>& mesh,
                                      const glm::vec3& color,
                                      const std::shared_ptr<Shader>& shader,
                                      const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Render a vao as lines
         * NEEDS to have an index buffer
         * 
         * @param vao The vertex array
         * @param color The color
         * @param shader The shader
         * @param camera The camera
         */
        static void drawLines(const std::shared_ptr<VertexArray>& vao,
                                     const glm::vec3& color,
                                     const std::shared_ptr<Shader>& shader,
                                     const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Render a vao as lines
         * NEEDS to have an index buffer
         * 
         * @param mesh The mesh
         * @param color The color
         * @param camera The camera
         */
        static void drawLines(const std::shared_ptr<Mesh>& mesh,
                                     const glm::vec3& color,
                                     const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Draw Circle
         * 
         * @param position The position
         * @param radius The radius
         * @param color The color
         * @param camera The camera
         */
        static void drawCircle(const glm::vec3& position,
                               const float& radius,
                               const glm::vec3& color,
                               const std::shared_ptr<Camera>& camera = {});

        /**
         * @brief Draw a grid
         * WARNING: CURRENTLY ONLY ONE GRID PER APPLICATION CAN BE RENDERED.
         * THE UNDERLYING GPU DATA WILL ALWAYS REPRESENT THE FIRST GRID THAT IS CALLED BY THIS FUNCTION EXCEPT IF reset = true
         * 
         * @param grid_dimension The spatial information about the grid
         * @param camera The camera
         * @param reset If the GPU data should be updated
         */ 
        static void drawGrid(const GridDimension& grid_dimension, const std::shared_ptr<Camera>& camera = {}, bool reset = false);

        /**
         * @brief Clear the currently bound framebuffer
         */
        static void clear();

        /**
         * @brief Destroys the renderer instance
         */
        inline static void destroy() { delete s_renderer; }

    private:
        Renderer();
        ~Renderer();

        class Impl;
        std::unique_ptr<Impl> impl;

        static Renderer* s_renderer;
    };
}