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
     *   @brief Initiliaze the context
     *   @param window The window for the context
     */
    void init(void* window);

    /**
     *   @brief Swap buffers in the swap chain
     */
    void swapBuffers();

private:
    void* _window;
};
}    // namespace atcg