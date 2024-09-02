#include <Renderer/Context.h>

#include <glad/glad.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <Core/Assert.h>
#include <Renderer/ContextData.h>

namespace atcg
{

namespace detail
{
static bool s_opengl_initialized = false;

void GLAPIENTRY MessageCallback(GLenum source,
                                GLenum type,
                                GLuint id,
                                GLenum severity,
                                GLsizei length,
                                const GLchar* message,
                                const void* userParam)
{
    switch(severity)
    {
        case GL_DEBUG_SEVERITY_LOW:
        case GL_DEBUG_SEVERITY_MEDIUM:
        {
            if(id == 131218) return;    // Some NVIDIA stuff going wrong -> disable this warning
            ATCG_WARN(message);
        }
        break;
        case GL_DEBUG_SEVERITY_HIGH:
        {
            ATCG_ERROR(message);
        }
        break;
        default:
            break;
    }
}
}    // namespace detail

void Context::destroy()
{
    if(_context_handle)
    {
        ContextData* data = (ContextData*)_context_handle;
        deactivate();
        eglDestroyContext(data->display, data->context);
        eglDestroySurface(data->display, data->surface);
        delete data;
        _context_handle = nullptr;
    }
}

void Context::create()
{
    ATCG_ASSERT(!_context_handle, "Handle already created");

    bool api = eglBindAPI(EGL_OPENGL_API);
    ATCG_ASSERT(api, "Could not bind opengl API");

    ContextData* data = new ContextData;
    data->display     = EGL_NO_DISPLAY;
    data->context     = EGL_NO_CONTEXT;
    data->surface     = EGL_NO_SURFACE;
    _context_handle   = (void*)data;

    data->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    ATCG_ASSERT(data->display != EGL_NO_DISPLAY, "Could not create EGL display");

    bool initialiazed = eglInitialize(data->display, nullptr, nullptr);
    ATCG_ASSERT(initialiazed, "Could not initialize EGL");

    // Choose EGL config
    EGLint configAttribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig eglConfig;
    EGLint numConfigs;
    bool choose = eglChooseConfig(data->display, configAttribs, &eglConfig, 1, &numConfigs);
    ATCG_ASSERT(choose, "Failed to choose display");

    // Create an EGL context
    EGLint contextAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6, EGL_NONE};
    data->context           = eglCreateContext(data->display, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    ATCG_ASSERT(data->context != EGL_NO_CONTEXT, "Failed to create context");

    // Make the context current
    eglMakeCurrent(data->display, data->surface, data->surface, data->context);

    makeCurrent();
}

void Context::initGraphicsAPI()
{
    if(!detail::s_opengl_initialized)
    {
        makeCurrent();

        if(!gladLoadGL())
        {
            ATCG_ERROR("Error loading glad!");
        }

#ifndef NDEBUG
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(detail::MessageCallback, 0);
#endif
        detail::s_opengl_initialized = true;
    }
}

void Context::swapBuffers()
{
    ContextData* data = (ContextData*)_context_handle;
    eglSwapBuffers(data->display, data->surface);
}

void Context::makeCurrent()
{
    ContextData* data = (ContextData*)_context_handle;
    eglMakeCurrent(data->display, data->surface, data->surface, data->context);
}

void Context::deactivate()
{
    ContextData* data = (ContextData*)_context_handle;
    eglMakeCurrent(data->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}
}    // namespace atcg