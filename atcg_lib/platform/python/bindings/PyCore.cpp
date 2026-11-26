#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Application, layers, window properties and events
void bindCore(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    py::class_<atcg::Application>(m, "Application");
    auto m_application = py::class_<PythonApplication, atcg::Application>(m, "PythonApplication");
    auto m_layer       = py::class_<atcg::Layer, PythonLayer, std::unique_ptr<atcg::Layer, py::nodelete>>(m, "Layer");

    auto m_event = py::class_<atcg::Event>(m, "Event");

    auto m_window_props        = py::class_<atcg::WindowProps>(m, "WindowProps");
    auto m_window_close_evnet  = py::class_<atcg::WindowCloseEvent, atcg::Event>(m, "WindowCloseEvent");
    auto m_window_resize_event = py::class_<atcg::WindowResizeEvent, atcg::Event>(m, "WindowResizeEvent");
    auto m_mouse_button_event  = py::class_<atcg::MouseButtonEvent, atcg::Event>(m, "MouseButtonEvent");
    auto m_mouse_button_pressed_event =
        py::class_<atcg::MouseButtonPressedEvent, atcg::MouseButtonEvent>(m, "MouseButtonPressedEvent");
    auto m_mouse_button_released_event =
        py::class_<atcg::MouseButtonReleasedEvent, atcg::MouseButtonEvent>(m, "MouseButtonReleasedEvent");
    auto m_mouse_moved_event     = py::class_<atcg::MouseMovedEvent, atcg::Event>(m, "MouseMovedEvent");
    auto m_mouse_scrolled_event  = py::class_<atcg::MouseScrolledEvent, atcg::Event>(m, "MouseScrolledEvent");
    auto m_key_event             = py::class_<atcg::KeyEvent, atcg::Event>(m, "KeyEvent");
    auto m_key_pressed_event     = py::class_<atcg::KeyPressedEvent, atcg::KeyEvent>(m, "KeyPressedEvent");
    auto m_key_released_event    = py::class_<atcg::KeyReleasedEvent, atcg::KeyEvent>(m, "KeyReleasedEvent");
    auto m_key_typed_event       = py::class_<atcg::KeyTypedEvent, atcg::KeyEvent>(m, "KeyTypedEvent");
    auto m_viewport_resize_event = py::class_<atcg::ViewportResizeEvent, atcg::Event>(m, "ViewportResizeEvent");
    auto m_asset_dropped_event   = py::class_<atcg::AssetDroppedEvent, atcg::Event>(m, "AssetDroppedEvent");

    ctx.define(
        [=]() mutable
        {
            m_window_props.def(py::init<>([]() { return atcg::WindowProps(); }))
                .def(py::init<const std::string&, uint32_t, uint32_t, int32_t, int32_t, bool>())
                .def_readwrite("tile", &atcg::WindowProps::title)
                .def_readwrite("width", &atcg::WindowProps::width)
                .def_readwrite("height", &atcg::WindowProps::height)
                .def_readwrite("pos_x", &atcg::WindowProps::pos_x)
                .def_readwrite("pos_y", &atcg::WindowProps::pos_y)
                .def_readwrite("vsync", &atcg::WindowProps::vsync)
                .def_readwrite("hidden", &atcg::WindowProps::hidden);

            m.def("init",
                  []()
                  {
                      atcg::WindowProps props;
                      props.hidden                           = true;
                      std::unique_ptr<PythonApplication> app = std::make_unique<PythonApplication>(props);
                      return app;
                  });

            m.def("shader_directory", []() { return atcg::shader_directory().string(); });
            m.def("resource_directory", []() { return atcg::resource_directory().string(); });

            m_application.def(py::init())
                .def(py::init<atcg::Layer*>())
                .def(py::init<atcg::WindowProps>())
                .def(py::init<atcg::Layer*, atcg::WindowProps>())
                .def("run", &atcg::Application::run);
            m_layer.def(py::init<>())
                .def(py::init<std::string>(), "name"_a)
                .def("onAttach", &atcg::Layer::onAttach)
                .def("onUpdate", &atcg::Layer::onUpdate, "delta_time"_a)
                .def("onImGuiRender", &atcg::Layer::onImGuiRender)
                .def("onEvent", &atcg::Layer::onEvent, "event"_a);

            m_event.def("getName", &atcg::Event::getName).def_readwrite("handled", &atcg::Event::handled);

            m_window_resize_event.def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
                .def("getWidth", &atcg::WindowResizeEvent::getWidth)
                .def("getHeight", &atcg::WindowResizeEvent::getHeight);
            m_mouse_button_event.def("getMouseButton", &atcg::MouseButtonEvent::getMouseButton)
                .def("getX", &atcg::MouseButtonEvent::getX)
                .def("getY", &atcg::MouseButtonEvent::getY);
            m_mouse_button_pressed_event.def(py::init<int32_t, float, float>(), "button"_a, "x"_a, "y"_a);
            m_mouse_button_released_event.def(py::init<int32_t, float, float>(), "button"_a, "x"_a, "y"_a);
            m_mouse_moved_event.def(py::init<float, float>(), "x"_a, "y"_a)
                .def("getX", &atcg::MouseMovedEvent::getX)
                .def("getY", &atcg::MouseMovedEvent::getY);
            m_mouse_scrolled_event.def(py::init<float, float>(), "x"_a, "y"_a)
                .def("getXOffset", &atcg::MouseScrolledEvent::getXOffset)
                .def("getYOffset", &atcg::MouseScrolledEvent::getYOffset);
            m_key_event.def("getKeyCode", &atcg::KeyEvent::getKeyCode);
            m_key_pressed_event.def(py::init<int32_t, bool>(), "key"_a, "key_pressed"_a)
                .def("isRepeat", &atcg::KeyPressedEvent::IsRepeat)
                .def("getCode", &atcg::KeyPressedEvent::getKeyCode);
            m_key_released_event.def(py::init<int32_t>());
            m_key_typed_event.def(py::init<int32_t>());
            m_viewport_resize_event.def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
                .def("getWidth", &atcg::ViewportResizeEvent::getWidth)
                .def("getHeight", &atcg::ViewportResizeEvent::getHeight);
            m_asset_dropped_event.def(py::init<atcg::AssetHandle, float, float>(), "handle"_a, "x"_a, "y"_a)
                .def("getHandle", &atcg::AssetDroppedEvent::getHandle)
                .def("getX", &atcg::AssetDroppedEvent::getX)
                .def("getY", &atcg::AssetDroppedEvent::getY);
            m.def("width",
                  []()
                  {
                      const auto& window = atcg::Application::get()->getWindow();
                      return (float)window->getWidth();
                  });

            m.def("height",
                  []()
                  {
                      const auto& window = atcg::Application::get()->getWindow();
                      return (float)window->getHeight();
                  });

            m.def("getViewportSize", []() { return atcg::Application::get()->getViewportSize(); });
            m.def("getViewportPosition", []() { return atcg::Application::get()->getViewportPosition(); });

            m.def(
                "enableDockSpace",
                [](bool enable) { atcg::Application::get()->enableDockSpace(enable); },
                "enable"_a);
        });
}
}    // namespace python
}    // namespace atcg
