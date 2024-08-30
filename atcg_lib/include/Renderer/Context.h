#pragma once

namespace atcg
{
/**
 *   @brief A class to model a graphcis context
 */
class Context
{
public:
    /**
     *   @brief Constructor
     */
    Context() = default;

    /**
     * @brief Destroy the context
     */
    void destroy();

    /**
     *   @brief Initiliaze the context
     */
    void initGraphicsAPI();

    /**
     * @brief Create the context
     */
    void create();

    /**
     *   @brief Swap buffers in the swap chain
     */
    void swapBuffers();

    /**
     * @brief Make this the current context for the thread
     */
    void makeCurrent();

    /**
     * @brief Deactivate this context for the current thread.
     * This function does not destroy the context.
     */
    void deactivate();

    /**
     * @brief The native context handle
     * For glfw backend it's the glfw window
     *
     * @return The handle
     */
    inline void* getContextHandle() const { return _context_handle; }

private:
    void* _context_handle = nullptr;
};
}    // namespace atcg