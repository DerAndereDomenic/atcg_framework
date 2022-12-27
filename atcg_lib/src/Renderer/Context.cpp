#include <Renderer/Context.h>

#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace atcg
{
    void Context::init(void* window)
    {
        glfwMakeContextCurrent((GLFWwindow*)window);

        if (!gladLoadGL())
        {
            std::cerr << "Error loading glad!\n";
        }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        _window = window;
    }

    void Context::swapBuffers()
    {
        glfwSwapBuffers((GLFWwindow*)_window);
    }
}