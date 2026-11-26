#include "Bindings.h"

using namespace pybind11::literals;

namespace atcg
{
namespace python
{
// Application, layers, window, input, logging, paths and events
void bindCore(BindingContext& ctx)
{
    py::module_ m = ctx.module();

    auto m_application_base = py::class_<atcg::Application>(m, "Application");
    auto m_application      = py::class_<PythonApplication, atcg::Application>(m, "PythonApplication");
    auto m_layer  = py::class_<atcg::Layer, PythonLayer, std::unique_ptr<atcg::Layer, py::nodelete>>(m, "Layer");
    auto m_window = py::class_<atcg::Window, std::unique_ptr<atcg::Window, py::nodelete>>(m, "Window");
    auto m_input  = py::class_<atcg::Input>(m, "Input");
    auto m_log    = m.def_submodule("Log");

    auto m_event_type     = py::enum_<atcg::EventType>(m, "EventType");
    auto m_event_category = py::enum_<atcg::EventCategory>(m, "EventCategory", py::arithmetic());
    auto m_event          = py::class_<atcg::Event>(m, "Event");

    auto m_window_props        = py::class_<atcg::WindowProps>(m, "WindowProps");
    auto m_window_close_evnet  = py::class_<atcg::WindowCloseEvent, atcg::Event>(m, "WindowCloseEvent");
    auto m_window_resize_event = py::class_<atcg::WindowResizeEvent, atcg::Event>(m, "WindowResizeEvent");
    auto m_file_dropped_event  = py::class_<atcg::FileDroppedEvent, atcg::Event>(m, "FileDroppedEvent");
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
            m_window_props
                .def(py::init<const std::string&,
                              uint32_t,
                              uint32_t,
                              int32_t,
                              int32_t,
                              bool,
                              bool,
                              const std::string&>(),
                     "title"_a     = "ATCG",
                     "width"_a     = 1600,
                     "height"_a    = 900,
                     "pos_x"_a     = std::numeric_limits<int32_t>::max(),
                     "pos_y"_a     = std::numeric_limits<int32_t>::max(),
                     "vsync"_a     = true,
                     "hidden"_a    = false,
                     "icon_path"_a = "")
                .def_readwrite("title", &atcg::WindowProps::title)
                .def_readwrite("width", &atcg::WindowProps::width)
                .def_readwrite("height", &atcg::WindowProps::height)
                .def_readwrite("pos_x", &atcg::WindowProps::pos_x)
                .def_readwrite("pos_y", &atcg::WindowProps::pos_y)
                .def_readwrite("vsync", &atcg::WindowProps::vsync)
                .def_readwrite("hidden", &atcg::WindowProps::hidden)
                .def_readwrite("icon_path", &atcg::WindowProps::icon_path);

            m.def("init",
                  []()
                  {
                      atcg::WindowProps props;
                      props.hidden                           = true;
                      std::unique_ptr<PythonApplication> app = std::make_unique<PythonApplication>(props);
                      return app;
                  });

            // Paths
            m.def("shader_directory", []() { return atcg::shader_directory().string(); });
            m.def("resource_directory", []() { return atcg::resource_directory().string(); });
            m.def("library_directory", []() { return atcg::library_directory().string(); });
            m.def("user_config_directory", []() { return atcg::user_config_directory().string(); });
            m.def("executable_path", []() { return atcg::executable_path().string(); });
            m.def("ptx_directory", []() { return atcg::ptx_directory().string(); });
            m.def("getCommandLineArguments", &atcg::getCommandLineArguments);

            // Logging through the engine logger
            m_log.def(
                "trace",
                [](const std::string& message) { ATCG_TRACE("{}", message); },
                "message"_a);
            m_log.def(
                "debug",
                [](const std::string& message) { ATCG_DEBUG("{}", message); },
                "message"_a);
            m_log.def(
                "info",
                [](const std::string& message) { ATCG_INFO("{}", message); },
                "message"_a);
            m_log.def(
                "warn",
                [](const std::string& message) { ATCG_WARN("{}", message); },
                "message"_a);
            m_log.def(
                "error",
                [](const std::string& message) { ATCG_ERROR("{}", message); },
                "message"_a);

            // Input polling
            m_input.def_static("isKeyPressed", &atcg::Input::isKeyPressed, "key"_a)
                .def_static("isKeyReleased", &atcg::Input::isKeyReleased, "key"_a)
                .def_static("isMouseButtonPressed", &atcg::Input::isMouseButtonPressed, "button"_a)
                .def_static("getMousePosition", &atcg::Input::getMousePosition);

#define ATCG_DEFINE_CODE(code) m.attr(#code) = static_cast<int32_t>(code)
            // Key and mouse codes as module attributes, e.g. ATCG_KEY_A
            ATCG_DEFINE_CODE(ATCG_KEY_UNKNOWN);
            ATCG_DEFINE_CODE(ATCG_KEY_SPACE);
            ATCG_DEFINE_CODE(ATCG_KEY_APOSTROPHE);
            ATCG_DEFINE_CODE(ATCG_KEY_COMMA);
            ATCG_DEFINE_CODE(ATCG_KEY_MINUS);
            ATCG_DEFINE_CODE(ATCG_KEY_PERIOD);
            ATCG_DEFINE_CODE(ATCG_KEY_SLASH);
            ATCG_DEFINE_CODE(ATCG_KEY_0);
            ATCG_DEFINE_CODE(ATCG_KEY_1);
            ATCG_DEFINE_CODE(ATCG_KEY_2);
            ATCG_DEFINE_CODE(ATCG_KEY_3);
            ATCG_DEFINE_CODE(ATCG_KEY_4);
            ATCG_DEFINE_CODE(ATCG_KEY_5);
            ATCG_DEFINE_CODE(ATCG_KEY_6);
            ATCG_DEFINE_CODE(ATCG_KEY_7);
            ATCG_DEFINE_CODE(ATCG_KEY_8);
            ATCG_DEFINE_CODE(ATCG_KEY_9);
            ATCG_DEFINE_CODE(ATCG_KEY_SEMICOLON);
            ATCG_DEFINE_CODE(ATCG_KEY_EQUAL);
            ATCG_DEFINE_CODE(ATCG_KEY_A);
            ATCG_DEFINE_CODE(ATCG_KEY_B);
            ATCG_DEFINE_CODE(ATCG_KEY_C);
            ATCG_DEFINE_CODE(ATCG_KEY_D);
            ATCG_DEFINE_CODE(ATCG_KEY_E);
            ATCG_DEFINE_CODE(ATCG_KEY_F);
            ATCG_DEFINE_CODE(ATCG_KEY_G);
            ATCG_DEFINE_CODE(ATCG_KEY_H);
            ATCG_DEFINE_CODE(ATCG_KEY_I);
            ATCG_DEFINE_CODE(ATCG_KEY_J);
            ATCG_DEFINE_CODE(ATCG_KEY_K);
            ATCG_DEFINE_CODE(ATCG_KEY_L);
            ATCG_DEFINE_CODE(ATCG_KEY_M);
            ATCG_DEFINE_CODE(ATCG_KEY_N);
            ATCG_DEFINE_CODE(ATCG_KEY_O);
            ATCG_DEFINE_CODE(ATCG_KEY_P);
            ATCG_DEFINE_CODE(ATCG_KEY_Q);
            ATCG_DEFINE_CODE(ATCG_KEY_R);
            ATCG_DEFINE_CODE(ATCG_KEY_S);
            ATCG_DEFINE_CODE(ATCG_KEY_T);
            ATCG_DEFINE_CODE(ATCG_KEY_U);
            ATCG_DEFINE_CODE(ATCG_KEY_V);
            ATCG_DEFINE_CODE(ATCG_KEY_W);
            ATCG_DEFINE_CODE(ATCG_KEY_X);
            ATCG_DEFINE_CODE(ATCG_KEY_Y);
            ATCG_DEFINE_CODE(ATCG_KEY_Z);
            ATCG_DEFINE_CODE(ATCG_KEY_LEFT_BRACKET);
            ATCG_DEFINE_CODE(ATCG_KEY_BACKSLASH);
            ATCG_DEFINE_CODE(ATCG_KEY_RIGHT_BRACKET);
            ATCG_DEFINE_CODE(ATCG_KEY_GRAVE_ACCENT);
            ATCG_DEFINE_CODE(ATCG_KEY_WORLD_1);
            ATCG_DEFINE_CODE(ATCG_KEY_WORLD_2);
            ATCG_DEFINE_CODE(ATCG_KEY_ESCAPE);
            ATCG_DEFINE_CODE(ATCG_KEY_ENTER);
            ATCG_DEFINE_CODE(ATCG_KEY_TAB);
            ATCG_DEFINE_CODE(ATCG_KEY_BACKSPACE);
            ATCG_DEFINE_CODE(ATCG_KEY_INSERT);
            ATCG_DEFINE_CODE(ATCG_KEY_DELETE);
            ATCG_DEFINE_CODE(ATCG_KEY_RIGHT);
            ATCG_DEFINE_CODE(ATCG_KEY_LEFT);
            ATCG_DEFINE_CODE(ATCG_KEY_DOWN);
            ATCG_DEFINE_CODE(ATCG_KEY_UP);
            ATCG_DEFINE_CODE(ATCG_KEY_PAGE_UP);
            ATCG_DEFINE_CODE(ATCG_KEY_PAGE_DOWN);
            ATCG_DEFINE_CODE(ATCG_KEY_HOME);
            ATCG_DEFINE_CODE(ATCG_KEY_END);
            ATCG_DEFINE_CODE(ATCG_KEY_CAPS_LOCK);
            ATCG_DEFINE_CODE(ATCG_KEY_SCROLL_LOCK);
            ATCG_DEFINE_CODE(ATCG_KEY_NUM_LOCK);
            ATCG_DEFINE_CODE(ATCG_KEY_PRINT_SCREEN);
            ATCG_DEFINE_CODE(ATCG_KEY_PAUSE);
            ATCG_DEFINE_CODE(ATCG_KEY_F1);
            ATCG_DEFINE_CODE(ATCG_KEY_F2);
            ATCG_DEFINE_CODE(ATCG_KEY_F3);
            ATCG_DEFINE_CODE(ATCG_KEY_F4);
            ATCG_DEFINE_CODE(ATCG_KEY_F5);
            ATCG_DEFINE_CODE(ATCG_KEY_F6);
            ATCG_DEFINE_CODE(ATCG_KEY_F7);
            ATCG_DEFINE_CODE(ATCG_KEY_F8);
            ATCG_DEFINE_CODE(ATCG_KEY_F9);
            ATCG_DEFINE_CODE(ATCG_KEY_F10);
            ATCG_DEFINE_CODE(ATCG_KEY_F11);
            ATCG_DEFINE_CODE(ATCG_KEY_F12);
            ATCG_DEFINE_CODE(ATCG_KEY_F13);
            ATCG_DEFINE_CODE(ATCG_KEY_F14);
            ATCG_DEFINE_CODE(ATCG_KEY_F15);
            ATCG_DEFINE_CODE(ATCG_KEY_F16);
            ATCG_DEFINE_CODE(ATCG_KEY_F17);
            ATCG_DEFINE_CODE(ATCG_KEY_F18);
            ATCG_DEFINE_CODE(ATCG_KEY_F19);
            ATCG_DEFINE_CODE(ATCG_KEY_F20);
            ATCG_DEFINE_CODE(ATCG_KEY_F21);
            ATCG_DEFINE_CODE(ATCG_KEY_F22);
            ATCG_DEFINE_CODE(ATCG_KEY_F23);
            ATCG_DEFINE_CODE(ATCG_KEY_F24);
            ATCG_DEFINE_CODE(ATCG_KEY_F25);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_0);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_1);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_2);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_3);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_4);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_5);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_6);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_7);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_8);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_9);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_DECIMAL);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_DIVIDE);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_MULTIPLY);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_SUBTRACT);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_ADD);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_ENTER);
            ATCG_DEFINE_CODE(ATCG_KEY_KP_EQUAL);
            ATCG_DEFINE_CODE(ATCG_KEY_LEFT_SHIFT);
            ATCG_DEFINE_CODE(ATCG_KEY_LEFT_CONTROL);
            ATCG_DEFINE_CODE(ATCG_KEY_LEFT_ALT);
            ATCG_DEFINE_CODE(ATCG_KEY_LEFT_SUPER);
            ATCG_DEFINE_CODE(ATCG_KEY_RIGHT_SHIFT);
            ATCG_DEFINE_CODE(ATCG_KEY_RIGHT_CONTROL);
            ATCG_DEFINE_CODE(ATCG_KEY_RIGHT_ALT);
            ATCG_DEFINE_CODE(ATCG_KEY_RIGHT_SUPER);
            ATCG_DEFINE_CODE(ATCG_KEY_MENU);
            ATCG_DEFINE_CODE(ATCG_KEY_LAST);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_1);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_2);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_3);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_4);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_5);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_6);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_7);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_8);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_LAST);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_LEFT);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_RIGHT);
            ATCG_DEFINE_CODE(ATCG_MOUSE_BUTTON_MIDDLE);
#undef ATCG_DEFINE_CODE

            // Window
            m_window.def("hide", &atcg::Window::hide)
                .def("show", &atcg::Window::show)
                .def("resize", &atcg::Window::resize, "width"_a, "height"_a)
                .def("toggleVSync", &atcg::Window::toggleVSync, "vsync"_a)
                .def("toggleFullscreen", &atcg::Window::toggleFullscreen)
                .def("getPosition", &atcg::Window::getPosition)
                .def("setPosition", &atcg::Window::setPosition, "position"_a)
                .def("isDecorated", &atcg::Window::isDecorated)
                .def("toggleDecoration", &atcg::Window::toogleDecoration)
                .def("getContentScale", &atcg::Window::getContentScale)
                .def("getWidth", &atcg::Window::getWidth)
                .def("getHeight", &atcg::Window::getHeight)
                .def("isFullScreen", &atcg::Window::isFullScreen)
                .def("isVSync", &atcg::Window::isVSync)
                .def("getContext", &atcg::Window::getContext)
                .def("getNativeWindow",
                     [](const atcg::Window& window) { return reinterpret_cast<uintptr_t>(window.getNativeWindow()); });

            // Application
            m_application_base.def_static("get", &atcg::Application::get, py::return_value_policy::reference)
                .def("run", &atcg::Application::run)
                .def("close", &atcg::Application::close)
                .def("pushLayer", &atcg::Application::pushLayer, "layer"_a, py::keep_alive<1, 2>())
                .def("onEvent", &atcg::Application::onEvent, "event"_a)
                .def(
                    "getWindow",
                    [](const atcg::Application& app) { return app.getWindow().get(); },
                    py::return_value_policy::reference_internal)
#ifndef ATCG_HEADLESS
                .def("getImGuiLayer", &atcg::Application::getImGuiLayer, py::return_value_policy::reference_internal)
#endif
                .def("getViewportSize", &atcg::Application::getViewportSize)
                .def("getViewportPosition", &atcg::Application::getViewportPosition)
                .def("enableDockSpace", &atcg::Application::enableDockSpace, "enable"_a)
                .def("getApplicationCounter", &atcg::Application::getApplicationCounter);

            m_application.def(py::init())
                .def(py::init<atcg::Layer*>(), "layer"_a, py::keep_alive<1, 2>())
                .def(py::init<atcg::WindowProps>(), "props"_a)
                .def(py::init<atcg::Layer*, atcg::WindowProps>(), "layer"_a, "props"_a, py::keep_alive<1, 2>());

            m_layer.def(py::init<>())
                .def(py::init<std::string>(), "name"_a)
                .def("onAttach", &atcg::Layer::onAttach)
                .def("onDetach", &atcg::Layer::onDetach)
                .def("onUpdate", &atcg::Layer::onUpdate, "delta_time"_a)
                .def("onImGuiRender", &atcg::Layer::onImGuiRender)
                .def("onEvent", &atcg::Layer::onEvent, "event"_a)
                .def("getName", &atcg::Layer::getName);

            // Events
            m_event_type.value("None", atcg::EventType::None)
                .value("WindowClose", atcg::EventType::WindowClose)
                .value("WindowResize", atcg::EventType::WindowResize)
                .value("WindowFocus", atcg::EventType::WindowFocus)
                .value("WindowLostFocus", atcg::EventType::WindowLostFocus)
                .value("WindowMoved", atcg::EventType::WindowMoved)
                .value("ViewportResize", atcg::EventType::ViewportResize)
                .value("KeyPressed", atcg::EventType::KeyPressed)
                .value("KeyReleased", atcg::EventType::KeyReleased)
                .value("KeyTyped", atcg::EventType::KeyTyped)
                .value("MouseButtonPressed", atcg::EventType::MouseButtonPressed)
                .value("MouseButtonReleased", atcg::EventType::MouseButtonReleased)
                .value("MouseMoved", atcg::EventType::MouseMoved)
                .value("MouseScrolled", atcg::EventType::MouseScrolled)
                .value("FileDropped", atcg::EventType::FileDropped)
                .value("AssetDropped", atcg::EventType::AssetDropped)
                .value("VRButtonPressed", atcg::EventType::VRButtonPressed)
                .value("VRButtonReleased", atcg::EventType::VRButtonReleased)
                .value("VRButtonTouched", atcg::EventType::VRButtonTouched)
                .value("VRButtonUntouched", atcg::EventType::VRButtonUntouched);

            m_event_category.value("None", atcg::EventCategory::None)
                .value("EventCategoryApplication", atcg::EventCategory::EventCategoryApplication)
                .value("EventCategoryInput", atcg::EventCategory::EventCategoryInput)
                .value("EventCategoryKeyboard", atcg::EventCategory::EventCategoryKeyboard)
                .value("EventCategoryMouse", atcg::EventCategory::EventCategoryMouse)
                .value("EventCategoryMouseButton", atcg::EventCategory::EventCategoryMouseButton)
                .value("EventCategoryVR", atcg::EventCategory::EventCategoryVR)
                .value("EventCategoryVRButton", atcg::EventCategory::EventCategroyVRButton);

            m_event.def("getName", &atcg::Event::getName)
                .def("getEventType", &atcg::Event::getEventType)
                .def("getCategoryFlags", &atcg::Event::getCategoryFlags)
                .def("isInCategory", &atcg::Event::isInCategory, "category"_a)
                .def("toString", &atcg::Event::toString)
                .def("__str__", [](const atcg::Event& event) { return event.toString(); })
                .def("__repr__", [](const atcg::Event& event) { return "<" + event.toString() + ">"; })
                .def_readwrite("handled", &atcg::Event::handled);

            m_window_close_evnet.def(py::init<>());
            m_window_resize_event.def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
                .def("getWidth", &atcg::WindowResizeEvent::getWidth)
                .def("getHeight", &atcg::WindowResizeEvent::getHeight);
            m_file_dropped_event.def("getCount", &atcg::FileDroppedEvent::getCount)
                .def(
                    "getPath",
                    [](const atcg::FileDroppedEvent& event, int i) -> py::object
                    {
                        if(i < 0 || i >= event.getCount())
                        {
                            ATCG_ERROR("FileDroppedEvent::getPath: Index {} out of range [0, {})", i, event.getCount());
                            return py::none();
                        }
                        return py::str(event.getPath(i));
                    },
                    "i"_a = 0)
                .def("getPaths",
                     [](const atcg::FileDroppedEvent& event)
                     {
                         std::vector<std::string> paths;
                         const char** raw_paths = event.getPaths();
                         for(int i = 0; i < event.getCount(); ++i)
                         {
                             paths.emplace_back(raw_paths[i]);
                         }
                         return paths;
                     });
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
            m_key_pressed_event.def(py::init<int32_t, bool>(), "key"_a, "is_repeat"_a = false)
                .def("isRepeat", &atcg::KeyPressedEvent::IsRepeat)
                .def("getCode", &atcg::KeyPressedEvent::getKeyCode);
            m_key_released_event.def(py::init<int32_t>(), "key"_a);
            m_key_typed_event.def(py::init<int32_t>(), "key"_a);
            m_viewport_resize_event.def(py::init<unsigned int, unsigned int>(), "width"_a, "height"_a)
                .def("getWidth", &atcg::ViewportResizeEvent::getWidth)
                .def("getHeight", &atcg::ViewportResizeEvent::getHeight);
            m_asset_dropped_event.def(py::init<atcg::AssetHandle, float, float>(), "handle"_a, "x"_a, "y"_a)
                .def("getHandle", &atcg::AssetDroppedEvent::getHandle)
                .def("getX", &atcg::AssetDroppedEvent::getX)
                .def("getY", &atcg::AssetDroppedEvent::getY);

            // Convenience functions on the active application
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
