#pragma once

#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Shader.h>
#include <Renderer/Camera.h>
#include <DataStructure/RenderMesh.h>

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
        inline static void init() { return s_renderer->initImpl(); }

        /**
         * @brief Set the clear color
         * 
         * @param color The clear color
         */
        inline static void setClearColor(const glm::vec4& color) {return s_renderer->clearColorImpl(color);}

        /**
         * @brief Change the viewport of the renderer
         * 
         * @param x The viewport x location
         * @param y The viewport y location
         * @param width The width
         * @param height The height
         */
        inline static void setViewport(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height) {s_renderer->setViewportImpl(x, y, width, height);}

        /**
         * @brief Render a vao
         * NEEDS to have an index buffer
         * 
         * @param vao The vertex array
         * @param shader The shader
         * @param camera The camera
         */
        inline static void draw(const std::shared_ptr<VertexArray>& vao,
                                const std::shared_ptr<Shader>& shader,
                                const std::shared_ptr<Camera>& camera = {}) {s_renderer->drawImpl(vao, shader, camera);}

        /**
         * @brief Render a mesh
         * 
         * @param mesh The mesh
         * @param shader The shader
         * @param camera The camera
         */
        inline static void draw(const std::shared_ptr<RenderMesh>& mesh,
                                const std::shared_ptr<Shader>& shader,
                                const std::shared_ptr<Camera>& camera = {}) {s_renderer->drawImpl(mesh, shader, camera);}

        /**
         * @brief Clear the currently bound framebuffer
         */
        inline static void clear() {return s_renderer->clearImpl();}

        /**
         * @brief Destroys the renderer instance
         */
        inline static void destroy() { delete s_renderer; }

    private:
        void initImpl();
        void clearColorImpl(const glm::vec4& color);
        void setViewportImpl(const uint32_t& x, const uint32_t& y, const uint32_t& width, const uint32_t& height);
        void drawImpl(const std::shared_ptr<VertexArray>& vao, const std::shared_ptr<Shader>& shader, const std::shared_ptr<Camera>& camera);
        void drawImpl(const std::shared_ptr<RenderMesh>& mesh, const std::shared_ptr<Shader>& shader, const std::shared_ptr<Camera>& camera);
        void clearImpl();

        static Renderer* s_renderer;
    };
}