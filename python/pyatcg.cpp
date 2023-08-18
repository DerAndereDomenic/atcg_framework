#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Core/EntryPoint.h>
#include <ATCG.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


class PythonLayer : public atcg::Layer
{
public:
    PythonLayer(const std::string& name = "Layer") : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override { PYBIND11_OVERRIDE(void, atcg::Layer, onAttach); }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override { PYBIND11_OVERRIDE(void, atcg::Layer, onUpdate, delta_time); }

    virtual void onImGuiRender() override { PYBIND11_OVERRIDE(void, atcg::Layer, onImGuiRender); }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override { PYBIND11_OVERRIDE(void, atcg::Layer, onEvent, event); }

private:
};

class PythonApplication : public atcg::Application
{
public:
    PythonApplication() : atcg::Application() {}

    ~PythonApplication() {}
};

atcg::Application* atcg::createApplication()
{
    return new PythonApplication;
}

int python_main(atcg::Layer* layer)
{
    atcg::Application* app = atcg::createApplication();
    app->pushLayer(layer);
    return atcg::atcg_main(app);
}

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, atcg::ref_ptr<T>);

PYBIND11_MODULE(pyatcg, m)
{
    m.doc() = R"pbdoc(
        Pybind11 atcg plugin
        -----------------------
        .. currentmodule:: pyatcg
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // ---------------- CORE ---------------------
    m.def("show", &python_main, "Start the application.");
    m.def("print_statistics", &atcg::print_statistics);

    py::class_<atcg::Layer, PythonLayer, std::unique_ptr<atcg::Layer, py::nodelete>>(m, "Layer")
        .def(py::init<>())
        .def("onAttach", &atcg::Layer::onAttach)
        .def("onUpdate", &atcg::Layer::onUpdate)
        .def("onImGuiRender", &atcg::Layer::onImGuiRender)
        .def("onEvent", &atcg::Layer::onEvent);

    py::class_<atcg::Event>(m, "Event")
        .def("getName", &atcg::Event::getName)
        .def_readwrite("handled", &atcg::Event::handled);
    py::class_<atcg::WindowCloseEvent, atcg::Event>(m, "WindowCloseEvent");
    py::class_<atcg::WindowResizeEvent, atcg::Event>(m, "WindowResizeEvent")
        .def(py::init<unsigned int, unsigned int>())
        .def("getWidth", &atcg::WindowResizeEvent::getWidth)
        .def("getHeight", &atcg::WindowResizeEvent::getHeight);
    py::class_<atcg::MouseButtonEvent, atcg::Event>(m, "MouseButtonEvent")
        .def("getMouseButton", &atcg::MouseButtonEvent::getMouseButton)
        .def("getX", &atcg::MouseButtonEvent::getX)
        .def("getY", &atcg::MouseButtonEvent::getY);
    py::class_<atcg::MouseButtonPressedEvent, atcg::MouseButtonEvent>(m, "MouseButtonPressedEvent")
        .def(py::init<int32_t, float, float>());
    py::class_<atcg::MouseButtonReleasedEvent, atcg::MouseButtonEvent>(m, "MouseButtonReleasedEvent")
        .def(py::init<int32_t, float, float>());
    py::class_<atcg::MouseMovedEvent, atcg::Event>(m, "MouseMovedEvent")
        .def(py::init<float, float>())
        .def("getX", &atcg::MouseMovedEvent::getX)
        .def("getY", &atcg::MouseMovedEvent::getY);
    py::class_<atcg::MouseScrolledEvent, atcg::Event>(m, "MouseScrolledEvent")
        .def(py::init<float, float>())
        .def("getXOffset", &atcg::MouseScrolledEvent::getXOffset)
        .def("getYOffset", &atcg::MouseScrolledEvent::getYOffset);
    py::class_<atcg::KeyEvent, atcg::Event>(m, "KeyEvent").def("getKeyCode", &atcg::KeyEvent::getKeyCode);
    py::class_<atcg::KeyPressedEvent, atcg::KeyEvent>(m, "KeyPressedEvent")
        .def(py::init<int32_t, bool>())
        .def("isRepeat", &atcg::KeyPressedEvent::IsRepeat)
        .def("getCode", &atcg::KeyPressedEvent::getCode);
    py::class_<atcg::KeyReleasedEvent, atcg::KeyEvent>(m, "KeyReleasedEvent").def(py::init<int32_t>());
    py::class_<atcg::KeyTypedEvent, atcg::KeyEvent>(m, "KeyTypedEvent").def(py::init<int32_t>());
    py::class_<atcg::ViewportResizeEvent, atcg::Event>(m, "ViewportResizeEvent")
        .def(py::init<unsigned int, unsigned int>())
        .def("getWidth", &atcg::ViewportResizeEvent::getWidth)
        .def("getHeight", &atcg::ViewportResizeEvent::getHeight);

    py::class_<atcg::Application, atcg::ref_ptr<atcg::Application>>(m, "Application");

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

    m.def("viewportSize()", []() { return atcg::Application::get()->getViewportSize(); });
    m.def("viewportPositions()", []() { return atcg::Application::get()->getViewportPosition(); });

    m.def("enableDockSpace", [](bool enable) { atcg::Application::get()->enableDockSpace(enable); });

    // ---------------- MATH -------------------------
    py::class_<glm::vec3>(m, "vec2", py::buffer_protocol())
        .def(py::init<float, float>())
        .def(py::init<float>())
        .def(py::init(
            [](py::array_t<float> b)
            {
                py::buffer_info info = b.request();

                // Copy for now, is there a better method?
                glm::vec2 v = glm::make_vec2(static_cast<float*>(info.ptr));

                return v;
            }))
        .def_buffer(
            [](glm::vec2& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {2},
                                       {sizeof(float)});
            });

    py::class_<glm::vec3>(m, "vec3", py::buffer_protocol())
        .def(py::init<float, float, float>())
        .def(py::init<float>())
        .def(py::init(
            [](py::array_t<float> b)
            {
                py::buffer_info info = b.request();

                // Copy for now, is there a better method?
                glm::vec3 v = glm::make_vec3(static_cast<float*>(info.ptr));

                return v;
            }))
        .def_buffer(
            [](glm::vec3& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {3},
                                       {sizeof(float)});
            });

    py::class_<glm::vec4>(m, "vec4", py::buffer_protocol())
        .def(py::init(
            [](py::array_t<float> b)
            {
                py::buffer_info info = b.request();

                // Copy for now, is there a better method?
                glm::vec4 v = glm::make_vec4(static_cast<float*>(info.ptr));

                return v;
            }))
        .def(py::init<float, float, float, float>())
        .def(py::init<float>())
        .def_buffer(
            [](glm::vec4& v) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(v),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       1,
                                       {4},
                                       {sizeof(float)});
            });

    py::class_<glm::mat3>(m, "mat3", py::buffer_protocol())
        .def(py::init(
            [](py::array_t<float> b)
            {
                py::buffer_info info = b.request();

                glm::mat3 M = glm::make_mat3(static_cast<float*>(info.ptr));

                return M;
            }))
        .def_buffer(
            [](glm::mat3& M) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(M),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       2,
                                       {3, 3},
                                       {sizeof(float), sizeof(float) * 3});
            });

    py::class_<glm::mat4>(m, "mat4", py::buffer_protocol())
        .def(py::init(
            [](py::array_t<float> b)
            {
                py::buffer_info info = b.request();

                glm::mat4 M = glm::make_mat4(static_cast<float*>(info.ptr));

                return M;
            }))
        .def_buffer(
            [](glm::mat4& M) -> py::buffer_info
            {
                return py::buffer_info(glm::value_ptr(M),
                                       sizeof(float),
                                       py::format_descriptor<float>::format(),
                                       2,
                                       {4, 4},
                                       {sizeof(float), sizeof(float) * 4});
            });

    // ------------------- Datastructure ---------------------------------
    py::class_<atcg::Graph, atcg::ref_ptr<atcg::Graph>>(m, "Graph").def(py::init<>());
    m.def("read_mesh", [](const std::string& path) { return atcg::IO::read_mesh(path); });

    py::class_<atcg::PerspectiveCamera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera")
        .def(py::init<>([](float aspect_ratio) { return atcg::make_ref<atcg::PerspectiveCamera>(aspect_ratio); }))
        .def("getPosition", &atcg::PerspectiveCamera::getPosition)
        .def("setPosition", &atcg::PerspectiveCamera::setPosition)
        .def("getView", &atcg::PerspectiveCamera::getView)
        .def("setView", &atcg::PerspectiveCamera::setView)
        .def("getProjection", &atcg::PerspectiveCamera::getProjection)
        .def("setProjection", &atcg::PerspectiveCamera::setProjection);

    py::class_<atcg::FirstPersonController, atcg::ref_ptr<atcg::FirstPersonController>>(m, "FirstPersonController")
        .def(py::init<>([](float aspect_ratio) { return atcg::make_ref<atcg::FirstPersonController>(aspect_ratio); }))
        .def("onUpdate", &atcg::FirstPersonController::onUpdate)
        .def("onEvent", &atcg::FirstPersonController::onEvent)
        .def("getCamera", &atcg::FirstPersonController::getCamera);

    // ------------------- Scene ---------------------------------
    py::class_<entt::entity>(m, "EntityHandle").def(py::init<uint32_t>());
    py::class_<atcg::Entity>(m, "Entity")
        .def(py::init<>())
        .def(py::init<entt::entity, atcg::Scene*>())
        .def("addTransformComponent",
             [](atcg::Entity& entity, const glm::vec3& position, const glm::vec3& scale, const glm::vec3& rotation)
             { return entity.addComponent<atcg::TransformComponent>(position, scale, rotation); })
        .def("addTransformComponent",
             [](atcg::Entity& entity, const atcg::TransformComponent& transform)
             { return entity.addComponent<atcg::TransformComponent>(transform); })
        .def("addGeometryComponent",
             [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Graph>& graph)
             { return entity.addComponent<atcg::GeometryComponent>(graph); })
        .def("addGeometryComponent",
             [](atcg::Entity& entity, const atcg::GeometryComponent& geometry)
             { return entity.addComponent<atcg::GeometryComponent>(geometry); })
        .def("addMeshRenderComponent",
             [](atcg::Entity& entity, const atcg::ref_ptr<atcg::Shader>& shader, const glm::vec3& color)
             { return entity.addComponent<atcg::MeshRenderComponent>(shader, color); })
        .def("addMeshRenderComponent",
             [](atcg::Entity& entity, const atcg::MeshRenderComponent& component)
             { return entity.addComponent<atcg::MeshRenderComponent>(component); })
        .def("addPointRenderComponent",
             [](atcg::Entity& entity,
                const atcg::ref_ptr<atcg::Shader>& shader,
                const glm::vec3& color,
                float point_size)
             { return entity.addComponent<atcg::PointRenderComponent>(shader, color, point_size); })
        .def("addPointRenderComponent",
             [](atcg::Entity& entity, const atcg::PointRenderComponent& component)
             { return entity.addComponent<atcg::PointRenderComponent>(component); })
        .def("addPointSphereRenderComponent",
             [](atcg::Entity& entity,
                const atcg::ref_ptr<atcg::Shader>& shader,
                const glm::vec3& color,
                float point_size)
             { return entity.addComponent<atcg::PointSphereRenderComponent>(shader, color, point_size); })
        .def("addPointSphereRenderComponent",
             [](atcg::Entity& entity, const atcg::PointSphereRenderComponent& component)
             { return entity.addComponent<atcg::PointSphereRenderComponent>(component); })
        .def("addEdgeRenderComponent",
             [](atcg::Entity& entity, const glm::vec3& color)
             { return entity.addComponent<atcg::EdgeRenderComponent>(color); })
        .def("addEdgeRenderComponent",
             [](atcg::Entity& entity, const atcg::EdgeRenderComponent& component)
             { return entity.addComponent<atcg::EdgeRenderComponent>(component); })
        .def("addEdgeCylinderRenderComponent",
             [](atcg::Entity& entity, const glm::vec3& color, float radius)
             { return entity.addComponent<atcg::EdgeCylinderRenderComponent>(color, radius); })
        .def("addEdgeCylinderRenderComponent",
             [](atcg::Entity& entity, const atcg::EdgeCylinderRenderComponent& component)
             { return entity.addComponent<atcg::EdgeCylinderRenderComponent>(component); })
        .def("hasTransformComponent", &atcg::Entity::hasComponent<atcg::TransformComponent>)
        .def("hasGeometryComponent", &atcg::Entity::hasComponent<atcg::GeometryComponent>)
        .def("hasMeshRenderComponent", &atcg::Entity::hasComponent<atcg::MeshRenderComponent>)
        .def("hasPointRenderComponent", &atcg::Entity::hasComponent<atcg::PointRenderComponent>)
        .def("hasPointSphereRenderComponent", &atcg::Entity::hasComponent<atcg::PointSphereRenderComponent>)
        .def("hasEdgeRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeRenderComponent>)
        .def("hasEdgeCylinderRenderComponent", &atcg::Entity::hasComponent<atcg::EdgeCylinderRenderComponent>);

    py::class_<atcg::Scene, atcg::ref_ptr<atcg::Scene>>(m, "Scene")
        .def(py::init<>([]() { return atcg::make_ref<atcg::Scene>(); }))
        .def("createEntity", &atcg::Scene::createEntity, py::arg("name") = "Entity");

    py::class_<atcg::TransformComponent>(m, "TransformComponent")
        .def(py::init<glm::vec3, glm::vec3, glm::vec3>())
        .def(py::init<glm::mat4>())
        .def("setPosition", &atcg::TransformComponent::setPosition)
        .def("setRotation", &atcg::TransformComponent::setRotation)
        .def("setScale", &atcg::TransformComponent::setScale)
        .def("setModel", &atcg::TransformComponent::setModel)
        .def("getPosition", &atcg::TransformComponent::getPosition)
        .def("getRotation", &atcg::TransformComponent::getRotation)
        .def("getScale", &atcg::TransformComponent::getScale)
        .def("getModel", &atcg::TransformComponent::getModel);

    py::class_<atcg::GeometryComponent>(m, "GeometryComponent")
        .def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Graph>&>())
        .def_readwrite("graph", &atcg::GeometryComponent::graph);

    py::class_<atcg::MeshRenderComponent>(m, "MeshRenderComponent")
        .def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&, glm::vec3>())
        .def_readwrite("visible", &atcg::MeshRenderComponent::visible)
        .def_readwrite("color", &atcg::MeshRenderComponent::color)
        .def_readwrite("shader", &atcg::MeshRenderComponent::shader);

    py::class_<atcg::PointRenderComponent>(m, "PointRenderComponent")
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&, glm::vec3, float>())
        .def_readwrite("visible", &atcg::PointRenderComponent::visible)
        .def_readwrite("color", &atcg::PointRenderComponent::color)
        .def_readwrite("shader", &atcg::PointRenderComponent::shader);

    py::class_<atcg::PointSphereRenderComponent>(m, "PointSphereRenderComponent")
        .def(py::init<const atcg::ref_ptr<atcg::Shader>&, glm::vec3, float>())
        .def_readwrite("visible", &atcg::PointSphereRenderComponent::visible)
        .def_readwrite("color", &atcg::PointSphereRenderComponent::color)
        .def_readwrite("shader", &atcg::PointSphereRenderComponent::shader);

    py::class_<atcg::EdgeRenderComponent>(m, "EdgeRenderComponent")
        .def(py::init<glm::vec3>())
        .def_readwrite("visible", &atcg::EdgeRenderComponent::visible)
        .def_readwrite("color", &atcg::EdgeRenderComponent::color);

    py::class_<atcg::EdgeCylinderRenderComponent>(m, "EdgeCylinderRenderComponent")
        .def(py::init<glm::vec3>())
        .def_readwrite("visible", &atcg::EdgeCylinderRenderComponent::visible)
        .def_readwrite("color", &atcg::EdgeCylinderRenderComponent::color);

    py::class_<atcg::SceneHierarchyPanel>(m, "SceneHierarchyPanel")
        .def(py::init<>())
        .def(py::init<const atcg::ref_ptr<atcg::Scene>&>())
        .def("renderPanel", &atcg::SceneHierarchyPanel::renderPanel)
        .def("selectEntity", &atcg::SceneHierarchyPanel::selectEntity)
        .def("getSelectedEntity", &atcg::SceneHierarchyPanel::getSelectedEntity);


    // ------------------- RENDERER ---------------------------------
    py::class_<atcg::Renderer>(m, "Renderer")
        .def_static("setClearColor", &atcg::Renderer::setClearColor)
        .def_static("clear", &atcg::Renderer::clear)
        .def("draw",
             [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
             { atcg::Renderer::draw(scene, camera); })
        .def("drawCADGrid",
             [](const atcg::ref_ptr<atcg::PerspectiveCamera>& camera) { atcg::Renderer::drawCADGrid(camera); })
        .def("drawCameras",
             [](const atcg::ref_ptr<atcg::Scene>& scene, const atcg::ref_ptr<atcg::PerspectiveCamera>& camera)
             { atcg::Renderer::drawCameras(scene, camera); });

    py::class_<atcg::Shader, atcg::ref_ptr<atcg::Shader>>(m, "Shader")
        .def(py::init<std::string, std::string>())
        .def(py::init<std::string, std::string, std::string>())
        .def("use", &atcg::Shader::use)
        .def("setInt", &atcg::Shader::setInt)
        .def("setFloat", &atcg::Shader::setFloat)
        .def("setVec3", &atcg::Shader::setVec3)
        .def("setVec4", &atcg::Shader::setVec4)
        .def("setMat4", &atcg::Shader::setMat4)
        .def("setMVP", &atcg::Shader::setMVP);

    py::class_<atcg::ShaderManager>(m, "ShaderManager")
        .def_static("getShader", &atcg::ShaderManager::getShader)
        .def_static("addShader", &atcg::ShaderManager::addShader)
        .def_static("addShaderFromName", &atcg::ShaderManager::addShaderFromName);

    // m.def("width",
    //       []()
    //       {
    //           const auto& window = atcg::Application::get()->getWindow();
    //           return (float)window->getWidth();
    //       });

    // m.def("height",
    //       []()
    //       {
    //           const auto& window = atcg::Application::get()->getWindow();
    //           return (float)window->getHeight();
    //       });

    // m.def("setSize",
    //       [](uint32_t width, uint32_t height)
    //       {
    //           const auto& window = atcg::Application::get()->getWindow();
    //           window->resize(width, height);
    //       });


    // py::class_<atcg::Input>(m, "Input")
    //     .def_static("isKeyPressed", &atcg::Input::isKeyPressed)
    //     .def_static("isMouseButtonPressed", &atcg::Input::isMouseButtonPressed)
    //     .def("getMousePosition",
    //          []()
    //          {
    //              glm::vec2 mouse_position = atcg::Input::getMousePosition();
    //              return py::array(2, reinterpret_cast<float*>(&mouse_position));
    //          });


    // py::class_<glm::vec3>(m, "Vector3", py::buffer_protocol())
    //     .def(py::init<float, float, float>())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             // Copy for now, is there a better method?
    //             glm::vec3 v;
    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 v = glm::make_vec3(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 v = glm::make_vec3(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 v = glm::make_vec3(static_cast<int*>(info.ptr));
    //             }

    //             return v;
    //         }))
    //     .def_buffer(
    //         [](glm::vec3& v) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(v),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    1,
    //                                    {3},
    //                                    {sizeof(float)});
    //         });

    // py::class_<glm::vec4>(m, "Vector4", py::buffer_protocol())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             // Copy for now, is there a better method?
    //             glm::vec4 v;
    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 v = glm::make_vec4(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 v = glm::make_vec4(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 v = glm::make_vec4(static_cast<int*>(info.ptr));
    //             }

    //             return v;
    //         }))
    //     .def_buffer(
    //         [](glm::vec4& v) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(v),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    1,
    //                                    {4},
    //                                    {sizeof(float)});
    //         });

    // py::class_<glm::mat3>(m, "Matrix3", py::buffer_protocol())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             glm::mat3 M;

    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 M = glm::make_mat3(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 M = glm::make_mat3(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 M = glm::make_mat3(static_cast<int*>(info.ptr));
    //             }

    //             return M;
    //         }))
    //     .def_buffer(
    //         [](glm::mat3& M) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(M),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    2,
    //                                    {3, 3},
    //                                    {sizeof(float), sizeof(float) * 3});
    //         });

    // py::class_<glm::mat4>(m, "Matrix4", py::buffer_protocol())
    //     .def(py::init(
    //         [](py::buffer b)
    //         {
    //             py::buffer_info info = b.request();

    //             glm::mat4 M;

    //             if(info.format == py::format_descriptor<float>::format())
    //             {
    //                 M = glm::make_mat4(static_cast<float*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<double>::format())
    //             {
    //                 M = glm::make_mat4(static_cast<double*>(info.ptr));
    //             }
    //             else if(info.format == py::format_descriptor<int>::format())
    //             {
    //                 M = glm::make_mat4(static_cast<int*>(info.ptr));
    //             }

    //             return glm::transpose(M);
    //         }))
    //     .def_buffer(
    //         [](glm::mat4& M) -> py::buffer_info
    //         {
    //             return py::buffer_info(glm::value_ptr(M),
    //                                    sizeof(float),
    //                                    py::format_descriptor<float>::format(),
    //                                    2,
    //                                    {4, 4},
    //                                    {sizeof(float), sizeof(float) * 4});
    //         });


    // py::class_<atcg::PerspectiveCamera, atcg::ref_ptr<atcg::PerspectiveCamera>>(m, "PerspectiveCamera")
    //     .def(py::init<float>())
    //     .def("getPosition", &atcg::PerspectiveCamera::getPosition)
    //     .def("setPosition", &atcg::PerspectiveCamera::setPosition)
    //     .def("getView", &atcg::PerspectiveCamera::getView)
    //     .def("setView", &atcg::PerspectiveCamera::setView)
    //     .def("getProjection", &atcg::PerspectiveCamera::getProjection)
    //     .def("setProjection", &atcg::PerspectiveCamera::setProjection);

    // py::class_<atcg::CameraController>(m, "CameraController")
    //     .def(py::init<float>())
    //     .def("onUpdate", &atcg::CameraController::onUpdate)
    //     .def("onEvent", &atcg::CameraController::onEvent)
    //     .def("getCamera", &atcg::CameraController::getCamera);

    // py::class_<atcg::Shader, atcg::ref_ptr<atcg::Shader>>(m, "Shader")
    //     .def(py::init<std::string, std::string>())
    //     .def(py::init<std::string, std::string, std::string>())
    //     .def("use", &atcg::Shader::use)
    //     .def("setInt", &atcg::Shader::setInt)
    //     .def("setFloat", &atcg::Shader::setFloat)
    //     .def("setVec3", &atcg::Shader::setVec3)
    //     .def("setVec4", &atcg::Shader::setVec4)
    //     .def("setMat4", &atcg::Shader::setMat4)
    //     .def("setMVP", &atcg::Shader::setMVP);

    // py::class_<atcg::ShaderManager>(m, "ShaderManager")
    //     .def_static("getShader", &atcg::ShaderManager::getShader)
    //     .def_static("addShader", &atcg::ShaderManager::addShader)
    //     .def_static("addShaderFromName", &atcg::ShaderManager::addShaderFromName);

    // py::class_<atcg::Mesh, atcg::ref_ptr<atcg::Mesh>>(m, "Mesh")
    //     .def("uploadData", &atcg::Mesh::uploadData)
    //     .def("setPosition", &atcg::Mesh::setPosition)
    //     .def("setScale", &atcg::Mesh::setScale)
    //     .def("setColor", &atcg::Mesh::setColor)
    //     .def("setColors", &atcg::Mesh::setColors)
    //     .def("requestVertexColors", &atcg::Mesh::request_vertex_colors)
    //     .def("requestVertexNormals", &atcg::Mesh::request_vertex_normals);
    // py::class_<atcg::PointCloud, atcg::ref_ptr<atcg::PointCloud>>(m, "PointCloud")
    //     .def("uploadData", &atcg::PointCloud::uploadData)
    //     .def("asMatrix", &atcg::PointCloud::asMatrix)
    //     .def("fromMatrix", &atcg::PointCloud::fromMatrix)
    //     .def("setColor", &atcg::PointCloud::setColor);

    // m.def("readMesh", &atcg::IO::read_mesh);
    // m.def("readPointCloud", &atcg::IO::read_pointcloud);

    // m.def("rayMeshIntersection", &atcg::Tracing::rayMeshIntersection);

    // py::enum_<atcg::DrawMode>(m, "DrawMode")
    //     .value("ATCG_DRAW_MODE_TRIANGLE", atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE)
    //     .value("ATCG_DRAW_MODE_POINTS", atcg::DrawMode::ATCG_DRAW_MODE_POINTS)
    //     .value("ATCG_DRAW_MODE_POINTS_SPHERE", atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE)
    //     .value("ATCG_DRAW_MODE_EDGES", atcg::DrawMode::ATCG_DRAW_MODE_EDGES);

    // py::class_<atcg::Renderer>(m, "Renderer")
    //     .def("init",
    //          [](uint32_t width, uint32_t height)
    //          {
    //              atcg::ref_ptr<atcg::Application> app = atcg::make_ref<atcg::Application>();
    //              const auto& window                   = app->getWindow();

    //              window->hide();
    //              window->resize(width, height);

    //              atcg::Renderer::useScreenBuffer();
    //              return app;
    //          })
    //     .def("setClearColor",
    //          [](const float r, const float g, const float b, const float a)
    //          { atcg::Renderer::setClearColor(glm::vec4(r, g, b, a)); })
    //     .def_static("setPointSize", &atcg::Renderer::setPointSize)
    //     .def_static("clear", &atcg::Renderer::clear)
    //     .def(
    //         "draw",
    //         [](const atcg::ref_ptr<atcg::Mesh>& mesh,
    //            const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
    //            const glm::vec3& color,
    //            const atcg::ref_ptr<atcg::Shader>& shader,
    //            atcg::DrawMode draw_mode) { atcg::Renderer::draw(mesh, camera, color, shader, draw_mode); },
    //         py::return_value_policy::automatic_reference)
    //     .def(
    //         "draw",
    //         [](const atcg::ref_ptr<atcg::PointCloud>& cloud,
    //            const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
    //            const glm::vec3& color,
    //            const atcg::ref_ptr<atcg::Shader>& shader)
    //         { atcg::Renderer::draw(cloud, camera, color, shader, atcg::DrawMode::ATCG_DRAW_MODE_POINTS); },
    //         py::return_value_policy::automatic_reference)
    //     .def("getFrame",
    //          []()
    //          {
    //              std::vector<uint8_t> buffer = atcg::Renderer::getFrame();
    //              return py::array(buffer.size(), buffer.data());
    //          })
    //     .def("getZBuffer",
    //          []()
    //          {
    //              std::vector<float> buffer = atcg::Renderer::getZBuffer();
    //              return py::array(buffer.size(), buffer.data());
    //          });

    // IMGUI BINDINGS

    m.def("BeginMainMenuBar", &ImGui::BeginMainMenuBar);
    m.def("EndMainMenuBar", &ImGui::EndMainMenuBar);
    m.def("BeginMenu", &ImGui::BeginMenu, py::arg("label"), py::arg("enabled") = true);
    m.def("EndMenu", &ImGui::EndMenu);
    m.def(
        "MenuItem",
        [](const char* label, const char* shortcut, bool* p_selected, bool enabled)
        {
            auto ret = ImGui::MenuItem(label, shortcut, p_selected, enabled);
            return std::make_tuple(ret, p_selected);
        },
        py::arg("label"),
        py::arg("shortcut"),
        py::arg("p_selected"),
        py::arg("enabled") = true,
        py::return_value_policy::automatic_reference);
    m.def(
        "Begin",
        [](const char* name, bool* p_open, ImGuiWindowFlags flags)
        {
            auto ret = ImGui::Begin(name, p_open, flags);
            return std::make_tuple(ret, p_open);
        },
        py::arg("name"),
        py::arg("p_open") = nullptr,
        py::arg("flags")  = 0,
        py::return_value_policy::automatic_reference);
    m.def("End", &ImGui::End);
    m.def(
        "Checkbox",
        [](const char* label, bool* v)
        {
            auto ret = ImGui::Checkbox(label, v);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::return_value_policy::automatic_reference);
    m.def(
        "Button",
        [](const char* label)
        {
            auto ret = ImGui::Button(label);
            return ret;
        },
        py::arg("label"),
        py::return_value_policy::automatic_reference);
    m.def(
        "SliderInt",
        [](const char* label, int* v, int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0)
        {
            auto ret = ImGui::SliderInt(label, v, v_min, v_max, format, flags);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::arg("v_min"),
        py::arg("v_max"),
        py::arg("format") = "%d",
        py::arg("flags")  = 0,
        py::return_value_policy::automatic_reference);
    m.def(
        "SliderFloat",
        [](const char* label,
           float* v,
           float v_min,
           float v_max,
           const char* format     = "%.3f",
           ImGuiSliderFlags flags = 0)
        {
            auto ret = ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
            return std::make_tuple(ret, v);
        },
        py::arg("label"),
        py::arg("v"),
        py::arg("v_min"),
        py::arg("v_max"),
        py::arg("format") = "%.3f",
        py::arg("flags")  = 0,
        py::return_value_policy::automatic_reference);
    m.def(
        "Text",
        [](const char* fmt)
        {
            ImGui::Text(fmt);
            return;
        },
        py::arg("fmt"),
        py::return_value_policy::automatic_reference);

    py::enum_<ImGuizmo::OPERATION>(m, "GuizmoOperation")
        .value("TRANSLATE", ImGuizmo::OPERATION::TRANSLATE)
        .value("ROTATE", ImGuizmo::OPERATION::ROTATE)
        .value("SCALE", ImGuizmo::OPERATION::SCALE)
        .export_values();
    m.def("drawGuizmo", atcg::drawGuizmo);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}