#pragma once

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
        inline static void clearColor(const glm::vec4& color) {return s_renderer->clearColorImpl(color);}

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
        void clearImpl();

        static Renderer* s_renderer;
    };
}