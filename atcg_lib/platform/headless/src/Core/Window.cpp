#include <Core/Window.h>
#include <Core/Assert.h>

#include <Renderer/Context.h>

namespace atcg
{

Window::Window(const WindowProps& props)
{
    _context = atcg::make_ref<Context>();

    _context->create();
    _context->initGraphicsAPI();

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
}

void Window::toggleVSync(bool vsync) {}

glm::vec2 Window::getPosition() const {}

float Window::getContentScale() const {}

void Window::hide() {}

void Window::show() {}
}    // namespace atcg