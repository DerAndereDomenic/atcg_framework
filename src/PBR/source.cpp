#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <random>
#include <stb_image.h>

class PBRLayer : public atcg::Layer
{
public:
    PBRLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

        auto skybox = atcg::IO::imread("res/pbr/skybox.hdr");
        ATCG_TRACE("{0} {1} {2}", skybox->width(), skybox->height(), skybox->channels());
        // atcg::Renderer::setSkybox(skybox);

        scene = atcg::IO::read_scene("res/cornell_box.obj");

        panel = atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>(scene);

        pt_panel = atcg::PathtracingPanel(scene);

        if(atcg::VRSystem::isVRAvailable())
        {
            float vr_aspect   = (float)atcg::VRSystem::width() / (float)atcg::VRSystem::height();
            camera_controller = atcg::make_ref<atcg::VRController>(vr_aspect);
            atcg::VRSystem::initControllerMeshes(scene);
        }
        else
        {
            const auto& window = atcg::Application::get()->getWindow();
            float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
            camera_controller  = atcg::make_ref<atcg::FirstPersonController>(aspect_ratio);
            camera_controller->getCamera()->setFOV(70.0f);
        }

        camera_controller->getCamera()->setPosition(glm::vec3(0.0f, 1.0f, 2.4f));
        camera_controller->getCamera()->setLookAt(glm::vec3(0.0f, 1.0f, 1.4f));
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(atcg::VRSystem::isVRAvailable())
        {
            atcg::ref_ptr<atcg::VRController> controller = camera_controller;

            if(controller->inMovement())
            {
                atcg::VRSystem::setMovementLine(controller->getControllerPosition(),
                                                controller->getControllerIntersection());
            }

            auto [t_left, t_right] = atcg::VRSystem::getRenderTargets();

            t_left->use();
            atcg::Renderer::setViewport(0, 0, atcg::VRSystem::width(), atcg::VRSystem::height());

            atcg::Renderer::clear();

            atcg::Renderer::draw(scene, controller->getCameraLeft());

            atcg::Renderer::drawCameras(scene, controller->getCameraLeft());

            atcg::Renderer::drawCADGrid(controller->getCameraLeft());

            if(controller->inMovement())
            {
                atcg::VRSystem::drawMovementLine(controller->getCameraLeft());
            }

            t_right->use();

            atcg::Renderer::clear();

            atcg::Renderer::draw(scene, controller->getCameraRight());

            atcg::Renderer::drawCameras(scene, controller->getCameraRight());

            atcg::Renderer::drawCADGrid(controller->getCameraRight());

            if(controller->inMovement())
            {
                atcg::VRSystem::drawMovementLine(controller->getCameraRight());
            }

            atcg::Renderer::useScreenBuffer();
            atcg::Renderer::setDefaultViewport();

            atcg::VRSystem::renderToScreen();
        }
        else
        {
            atcg::Renderer::clear();

            atcg::Renderer::draw(scene, camera_controller->getCamera());

            atcg::Renderer::drawCameras(scene, camera_controller->getCamera());

            atcg::Renderer::drawCADGrid(camera_controller->getCamera());
        }
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("File"))
        {
            if(ImGui::MenuItem("Save"))
            {
                atcg::Serializer<atcg::ComponentSerializer> serializer(scene);

                serializer.serialize("../Scene/Scene.json");
            }

            if(ImGui::MenuItem("Load"))
            {
                scene->removeAllEntites();
                atcg::Serializer<atcg::ComponentSerializer> serializer(scene);

                serializer.deserialize("../Scene/Scene.json");

                hovered_entity = atcg::Entity();
                panel.selectEntity(hovered_entity);
            }

            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            ImGui::End();
        }
        pt_panel.renderPanel(camera_controller->getCamera());

        panel.renderPanel();
        hovered_entity = panel.getSelectedEntity();

        atcg::drawGuizmo(hovered_entity, current_operation, camera_controller->getCamera());
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onKeyPressed));
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);
        return false;
    }

    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == GLFW_KEY_T)
        {
            current_operation = ImGuizmo::OPERATION::TRANSLATE;
        }
        if(event->getKeyCode() == GLFW_KEY_R)
        {
            current_operation = ImGuizmo::OPERATION::ROTATE;
        }
        if(event->getKeyCode() == GLFW_KEY_S)
        {
            current_operation = ImGuizmo::OPERATION::SCALE;
        }
        // if(event->getKeyCode() == GLFW_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        if(event->getKeyCode() == GLFW_KEY_P)
        {
            auto imgui_layer = atcg::Application::get()->getImGuiLayer();
            imgui_layer->setPathtracingFocus();
            atcg::Pathtracer::draw(scene,
                                   camera_controller->getCamera(),
                                   atcg::RaytracingShaderManager::getShader("Pathtracing"),
                                   imgui_layer->getViewportSize().x,
                                   imgui_layer->getViewportSize().y,
                                   1024);
        }

        return true;
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(in_viewport && event->getMouseButton() == GLFW_MOUSE_BUTTON_LEFT && !ImGuizmo::IsOver())
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1 ? atcg::Entity() : atcg::Entity((entt::entity)id, scene.get());
            panel.selectEntity(hovered_entity);
        }
        return true;
    }

    bool onMouseMoved(atcg::MouseMovedEvent* event)
    {
        const atcg::Application* app = atcg::Application::get();
        glm::ivec2 offset            = app->getViewportPosition();
        int height                   = app->getViewportSize().y;
        mouse_pos                    = glm::vec2(event->getX() - offset.x, height - (event->getY() - offset.y));

        in_viewport =
            mouse_pos.x >= 0 && mouse_pos.y >= 0 && mouse_pos.y < height && mouse_pos.x < app->getViewportSize().x;

        return false;
    }

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity hovered_entity;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::ref_ptr<atcg::Graph> plane;

    atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler> panel;
    atcg::PathtracingPanel pt_panel;

    float time       = 0.0f;
    bool in_viewport = false;

    glm::vec2 mouse_pos;

    bool show_render_settings             = false;
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
};

class PBR : public atcg::Application
{
public:
    PBR(const atcg::WindowProps& props) : atcg::Application(props) { pushLayer(new PBRLayer("Layer")); }

    ~PBR() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = true;
    return new PBR(props);
}