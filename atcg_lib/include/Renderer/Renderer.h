#pragma once

#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Shader.h>
#include <Renderer/PerspectiveCamera.h>
#include <DataStructure/Mesh.h>

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
         */
        static void init();

        /**
         * @brief Set the clear color
         * 
         * @param color The clear color
         */
        static void setClearColor(const glm::vec4& color);

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