#include <Renderer/Context.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Core/Assert.h>

namespace atcg
{

namespace detail
{
static void GLFWErrorCallback(int error, const char* description)
{
    ATCG_ERROR("GLFW Error: {0}: {1}", error, description);
}

}    // namespace detail

void Context::destroy()
{
    if(_context_handle)
    {
        deactivate();
        glfwDestroyWindow((GLFWwindow*)_context_handle);
    }
}

void Context::create(const int device_id)
{
    ATCG_ASSERT(!_context_handle, "Context already created");

    if(device_id != 0)
    {
        ATCG_WARN("A device id that is not 0 ({0}) was requested for glfw backend. This is unsupported", device_id);
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    _context_handle = (void*)glfwCreateWindow(1, 1, "ATCG", nullptr, nullptr);
    ATCG_ASSERT(_context_handle, "Could not create context");
    makeCurrent();
}

void Context::create(const atcg::ref_ptr<Context>& shared)
{
    ATCG_ASSERT(!_context_handle, "Context already created");

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* glfwShared = nullptr;

    if(shared)
    {
        shared->deactivate();
        glfwShared = (GLFWwindow*)shared->getContextHandle();
    }

    _context_handle = (void*)glfwCreateWindow(1, 1, "ATCG", nullptr, glfwShared);
    ATCG_ASSERT(_context_handle, "Could not create context");
    makeCurrent();
}

void Context::swapBuffers()
{
    glfwSwapBuffers((GLFWwindow*)_context_handle);
}

void Context::makeCurrent()
{
    glfwMakeContextCurrent((GLFWwindow*)_context_handle);
}

void Context::deactivate()
{
    glfwMakeContextCurrent(NULL);
}

bool Context::isCurrent() const
{
    return glfwGetCurrentContext() == ((GLFWwindow*)_context_handle);
}

ContextHandle Context::getCurrentContextHandle()
{
    return (ContextHandle)glfwGetCurrentContext();
}
}    // namespace atcg