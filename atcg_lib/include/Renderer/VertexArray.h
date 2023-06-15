#pragma once

#include <Renderer/Buffer.h>
#include <Core/Memory.h>

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
     * @brief Push a vertex buffer to the vbo stack
     *
     * @param vbo The vertex buffer to add
     */
    void pushVertexBuffer(const atcg::ref_ptr<VertexBuffer>& vbo);

    /**
     * @brief Pop a vertex buffer from the vbo stack
     *
     * @return The Vertex buffer that was removed
     */
    const atcg::ref_ptr<VertexBuffer>& popVertexBuffer();

    /**
     * @brief Get the top vertex buffer on the stack
     *
     * @return The vertex buffer
     */
    const atcg::ref_ptr<VertexBuffer>& peekVertexBuffer() const;

    /**
     * @brief Set the Index Buffer
     *
     * @param ibo The index buffer
     */
    void setIndexBuffer(const atcg::ref_ptr<IndexBuffer>& ibo);

    /**
     * @brief Set an instance buffer used for instance rendering
     *
     * @param buffer The buffer
     */
    void pushInstanceBuffer(const atcg::ref_ptr<VertexBuffer>& vbo);

    /**
     * @brief Get the Index Buffer object
     *
     * @return const atcg::ref_ptr<IndexBuffer>& The index buffer
     */
    inline const atcg::ref_ptr<IndexBuffer>& getIndexBuffer() const { return _ibo; }

private:
    /**
     * @brief Changes how the underlying vertex buffer is interpreted for instance rendering.
     * It's always applied on the top element of the stack
     *
     * @param divisor When the buffer should be updated (default = 0, it is updated on every iteration of the vertex
     * shader). Set to 1 for one update per instance.
     */
    void markInstance(uint32_t divisor);

private:
    struct VertexBufferIndexing
    {
        uint32_t vertex_buffer_index;
        uint32_t divisor = 0;
    };

    uint32_t _ID;
    // This vector stores the last index buffer (exclusive) that is used by the i-th vertex buffer. For example: If the
    // first vbo has attributes 0,1,2, the first element of this vector will be 3. If the second vbo manages 3 and 4, it
    // will store a 5 and so on.
    std::vector<VertexBufferIndexing> _vertex_buffer_index = {};    // Stack
    std::vector<atcg::ref_ptr<VertexBuffer>> _vertex_buffers;       // Stack
    atcg::ref_ptr<IndexBuffer> _ibo;
};
}    // namespace atcg