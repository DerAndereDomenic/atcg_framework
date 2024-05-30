#pragma once

#include <Core/Memory.h>
#include <Renderer/Buffer.h>

namespace atcg
{
/**
 * @brief A class to model a VertexArray
 */
class VertexArray
{
public:
    /**
     * @brief Construct a new VertexArray object
     */
    VertexArray();

    /**
     * @brief Destroy the VertexArray object
     */
    ~VertexArray();

    /**
     * @brief Use this vao
     */
    void use() const;

    /**
     * @brief Push a VertexBuffer to the vbo stack
     *
     * @param vbo The VertexBuffer to add
     */
    void pushVertexBuffer(const atcg::ref_ptr<VertexBuffer>& vbo);

    /**
     * @brief Pop a VertexBuffer from the vbo stack
     *
     * @return The VertexBuffer that was removed
     */
    atcg::ref_ptr<VertexBuffer> popVertexBuffer();

    /**
     * @brief Get the top VertexBuffer on the stack
     *
     * @return The VertexBuffer
     */
    const atcg::ref_ptr<VertexBuffer>& peekVertexBuffer() const;

    /**
     * @brief Set the IndexBuffer
     *
     * @param ibo The IndexBuffer
     */
    void setIndexBuffer(const atcg::ref_ptr<IndexBuffer>& ibo);

    /**
     * @brief Set an instance buffer used for instance rendering
     *
     * @param buffer The buffer
     */
    void pushInstanceBuffer(const atcg::ref_ptr<VertexBuffer>& vbo);

    /**
     * @brief Get the IndexBuffer object
     *
     * @return const atcg::ref_ptr<IndexBuffer>& The IndexBuffer
     */
    ATCG_INLINE const atcg::ref_ptr<IndexBuffer>& getIndexBuffer() const { return _ibo; }

private:
    /**
     * @brief Changes how the underlying VertexBuffer is interpreted for instance rendering.
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
    // This vector stores the last IndexBuffer (exclusive) that is used by the i-th VertexBuffer. For example: If the
    // first vbo has attributes 0,1,2, the first element of this vector will be 3. If the second vbo manages 3 and 4, it
    // will store a 5 and so on.
    std::vector<VertexBufferIndexing> _vertex_buffer_index = {};    // Stack
    std::vector<atcg::ref_ptr<VertexBuffer>> _vertex_buffers;       // Stack
    atcg::ref_ptr<IndexBuffer> _ibo;
};
}    // namespace atcg