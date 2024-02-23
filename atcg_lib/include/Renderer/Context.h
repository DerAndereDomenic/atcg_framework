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
     * @brief Initialize the windowing API for the current module
     */
    void initWindowingAPI();

    /**
     * @brief Deinit the windowing API
     */
    void deinitWindowingAPI();

    /**
     *   @brief Initiliaze the context
     *   @param window The window for the context
     */
    void initGraphicsAPI(void* window);

    /**
     *   @brief Swap buffers in the swap chain
     */
    void swapBuffers();

private:
    void* _window;
};
}    // namespace atcg