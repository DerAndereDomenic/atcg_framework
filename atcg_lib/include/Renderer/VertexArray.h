#pragma once

#include <Renderer/Buffer.h>
#include <memory>

namespace atcg
{
    /**
     * @brief A class to model a vertex array
     */
    class VertexArray
    {
    public:
        /**
         * @brief Construct a new Vertex Array object
         */
        VertexArray();

        /**
         * @brief Destroy the Vertex Array object
         */
        ~VertexArray();

        /**
         * @brief Use this vao
         */
        void use() const;

        /**
         * @brief Add a vertex buffer
         * 
         * @param vbo The vertex buffer to add
         */
        void addVertexBuffer(const std::shared_ptr<VertexBuffer>& vbo);

        /**
         * @brief Set the Index Buffer
         * 
         * @param ibo The index buffer
         */
        void setIndexBuffer(const std::shared_ptr<IndexBuffer>& ibo);

        /**
         * @brief Get the Vertex Buffers object
         * 
         * @return const std::vector<std::shared_ptr<VertexBuffer>>& The vertex buffers
         */
        inline const std::vector<std::shared_ptr<VertexBuffer>>& getVertexBuffers() const {return _vertex_buffers;}

        /**
         * @brief Get the Index Buffer object
         * 
         * @return const std::shared_ptr<IndexBuffer>& The index buffer
         */
        inline const std::shared_ptr<IndexBuffer>& getIndexBuffer() const {return _ibo;}
    private:
        uint32_t _ID;
        uint32_t _vertex_buffer_index = 0;
        std::vector<std::shared_ptr<VertexBuffer>> _vertex_buffers;
        std::shared_ptr<IndexBuffer> _ibo;
    };
}