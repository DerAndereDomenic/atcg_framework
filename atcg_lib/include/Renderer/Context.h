#pragma once

#include <Core/Memory.h>

namespace atcg
{
class ContextManagerSystem;
class Framebuffer;
using ContextHandle = uint64_t;

/**
 *   @brief A class to model a graphcis context
 */
class Context
{
public:
    /**
     *   @brief Initiliaze the context
     */
    void initGraphicsAPI();

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
     * @brief Check if the given context is current
     *
     * @return True if the context is current
     */
    bool isCurrent() const;

    /**
     * @brief Get the current fbo of this context
     *
     * @return The ID of the fbo
     */
    ATCG_INLINE uint32_t getCurrentFBO() const { return _fbo; }

    /**
     * @brief The native context handle
     * For glfw backend it's the glfw window
     *
     * @return The handle
     */
    ATCG_INLINE ContextHandle getContextHandle() const { return (ContextHandle)_context_handle; }

    /**
     * @brief Get the handle of the current context
     *
     * @return The current context handle
     */
    static ContextHandle getCurrentContextHandle();

private:
    /**
     *   @brief Constructor
     */
    Context() = default;

    /**
     * @brief Destroy the context
     */
    void destroy();

    /**
     * @brief Create the context
     * @note After creation this context will be the current context. Therefore, it is assumed that no context is
     * associated with this thread when this function is called.
     *
     * @param device_id The device id on which the context should be created
     * @note This is only used for headless rendering (on linux). For normal in-window rendering, this value is ignored
     */
    void create(const int device_id = 0);

    /**
     * @brief Create the context
     * This function is used to create a shared context.
     * @note After creation this context will be the current context. The device of this context will be the same as the
     * shared context.
     *
     * @param shared The context to share from
     */
    void create(const atcg::ref_ptr<Context>& shared);

    /**
     * @brief Set the currently bound fbo
     *
     * @param fbo The ID of the fbo
     */
    ATCG_INLINE void setCurrentFBO(const uint32_t fbo) { _fbo = fbo; }

    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

private:
    void* _context_handle = nullptr;
    uint32_t _fbo         = 0;

    friend class ContextManagerSystem;
    friend class Framebuffer;
};
}    // namespace atcg