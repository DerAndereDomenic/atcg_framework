#include <Core/Window.h>
#include <Core/Assert.h>

#include <Renderer/Context.h>
#include <Renderer/ContextData.h>

namespace atcg
{

Window::Window(const WindowProps& props)
{
    _context = atcg::make_ref<Context>();

    _context->create();
    _context->initGraphicsAPI();

    EGLint pbufferAttribs[] = {EGL_WIDTH,
                               (int)props.width,
                               EGL_HEIGHT,
                               (int)props.height,
                               EGL_TEXTURE_FORMAT,
                               EGL_TEXTURE_RGBA,
                               EGL_TEXTURE_TARGET,
                               EGL_TEXTURE_2D,
                               EGL_MIPMAP_TEXTURE,
                               EGL_TRUE,
                               EGL_NONE};
    EGLConfig eglConfig;
    ContextData* context_data = (ContextData*)_context->getContextHandle();
    context_data->surface     = eglCreatePbufferSurface(context_data->display, eglConfig, pbufferAttribs);

    _data.width  = props.width;
    _data.height = props.height;
}

Window::~Window()
{
    _context->destroy();
}

void Window::onUpdate()
{
    ATCG_ASSERT(_context, "No valid context");

    _context->swapBuffers();
}

void Window::setEventCallback(const EventCallbackFn& callback)
{
    _data.on_event = callback;
}

void* Window::getNativeWindow() const
{
    return _context->getContextHandle();
}

void Window::resize(const uint32_t& _width, const uint32_t& _height)
{
    _data.width = _width, _data.height = _height;

    EGLint pbufferAttribs[] = {EGL_WIDTH,
                               (int)_width,
                               EGL_HEIGHT,
                               (int)_height,
                               EGL_TEXTURE_FORMAT,
                               EGL_TEXTURE_RGBA,
                               EGL_TEXTURE_TARGET,
                               EGL_TEXTURE_2D,
                               EGL_MIPMAP_TEXTURE,
                               EGL_TRUE,
                               EGL_NONE};
    EGLConfig eglConfig;
    ContextData* context_data = (ContextData*)_context->getContextHandle();

    if(context_data->surface != EGL_NO_SURFACE)
    {
        eglDestroySurface(context_data->display, context_data->surface);
    }

    context_data->surface = eglCreatePbufferSurface(context_data->display, eglConfig, pbufferAttribs);
}

void Window::toggleVSync(bool vsync) {}

void Window::toggleFullscreen() {}

glm::vec2 Window::getPosition() const
{
    return glm::vec2(0);
}

void Window::setPosition(const glm::vec2& position) {}

bool Window::isDecorated() const
{
    return false;
}

void Window::toogleDecoration() {}

float Window::getContentScale() const
{
    return 1.0f;
}

void Window::hide() {}

void Window::show() {}
}    // namespace atcg