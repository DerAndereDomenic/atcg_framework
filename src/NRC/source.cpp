#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>
#include <portable-file-dialogs.h>

#include <Core/Common.h>

#include <Integrator/IntegratorRegistry.h>

#include "NRCIntegrator.h"

class NRCLayer : public atcg::Layer
{
public:
    NRCLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);

        atcg::Project::load("../CornellBox4/Project.json");

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        atcg::CameraIntrinsics intrinsics;
        intrinsics.setAspectRatio(aspect_ratio);
        atcg::CameraExtrinsics extrinsics;
        extrinsics.setPosition(glm::vec3(0, 1, 3));
        extrinsics.setTarget(glm::vec3(0, 1, 0));
        camera_controller = atcg::make_ref<atcg::FirstPersonController>(
            atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics));


        atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());

        // The NRC integrator is local to this application, so it is registered here to be usable by the controller
        ATCG_REGISTER_INTEGRATOR(atcg::IntegratorRegistry::getRegistry(),
                                 atcg::NRCIntegrator::getType(),
                                 atcg::NRCIntegrator);

        integrator_controller = atcg::make_ref<atcg::IntegratorController>();
        integrator_panel      = atcg::GUI::IntegratorPanel(integrator_controller);
        integrator_controller->setIntegratorType(atcg::NRCIntegrator::getType());
        integrator_controller->start();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        performance_panel.registerFrameTime(delta_time);
        camera_controller->onUpdate(delta_time);

        atcg::Scripting::handleScriptUpdates(atcg::Project::getActive()->getActiveScene(), delta_time);

        // The controller rebuilds or resets the integrator itself if the scene, camera or viewport changed
        auto scene = atcg::Project::getActive()->getActiveScene();
        if(!integrator_controller->render(scene, atcg::Renderer::getFramebuffer()))
        {
            atcg::SceneRenderer::render(scene, camera_controller->getCamera(), atcg::Renderer::getFramebuffer());

            atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());
            atcg::Renderer::drawCADGrid(camera_controller->getCamera());
            atcg::GraphicsCommand::endRenderPass();
        }
    }

#ifndef ATCG_HEADLESS
    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("File"))
        {
            if(ImGui::MenuItem("New"))
            {
                atcg::Project::create("DefaultProject");
                atcg::AssetManager::clear();
                atcg::RevisionStack::clearChache();

                atcg::Project::getActive()->setActiveScene(atcg::make_ref<atcg::Scene>());
                atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());
                // This application always uses the NRC integrator
                integrator_controller->setIntegratorType(atcg::NRCIntegrator::getType());
                saved = false;
            }

            if(ImGui::MenuItem("Save", (const char*)0, false, saved))
            {
                atcg::Project::getActive()->save();
            }

            if(ImGui::MenuItem("Save as..."))
            {
                auto f     = pfd::save_file("Choose project location", pfd::path::home(), {}, true);
                auto files = f.result();

                if(!files.empty())
                {
                    auto path = std::filesystem::path(files);
                    atcg::Project::saveActive(path);

                    saved = true;
                }
            }

            if(ImGui::MenuItem("Load"))
            {
                auto f     = pfd::open_file("Choose project file",
                                            pfd::path::home(),
                                            {"Project file (.json)", "*.json"},
                                            pfd::opt::none);
                auto files = f.result();

                if(!files.empty())
                {
                    atcg::Project::load(files[0]);

                    if(atcg::Project::getActive()->getActiveScene())
                    {
                        atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());
                    }
                    // This application always uses the NRC integrator
                    integrator_controller->setIntegratorType(atcg::NRCIntegrator::getType());
                    atcg::RevisionStack::clearChache();

                    saved = true;
                }
            }

            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Debug"))
        {
            ImGui::MenuItem("Show Performance Panel", nullptr, &show_performance);
            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::MenuItem("Show Integrator Settings", nullptr, &show_integrator_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        render_settings_panel.renderPanel(show_render_settings);
        integrator_panel.renderPanel(show_integrator_settings);

        performance_panel.renderPanel(show_performance);
        panel.renderPanel(atcg::Project::getActive()->getActiveScene());

        asset_panel.renderPanel();

        atcg::drawGuizmo(atcg::Project::getActive()->getActiveScene(),
                         current_operation,
                         atcg::GuizmoMode::LOCAL,
                         camera_controller->getCamera());
    }
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::Scripting::handleScriptEvents(atcg::Project::getActive()->getActiveScene(), event);

        atcg::EventDispatcher dispatcher(event);
#ifndef ATCG_HEADLESS
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(NRCLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(NRCLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(NRCLayer::onKeyPressed));
#endif
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(NRCLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);
        return false;
    }

#ifndef ATCG_HEADLESS
    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == ATCG_KEY_T)
        {
            current_operation = atcg::GuizmoOperation::TRANSLATE;
        }
        if(event->getKeyCode() == ATCG_KEY_R)
        {
            if(atcg::Input::isKeyPressed(ATCG_KEY_LEFT_CONTROL))
            {
                atcg::Scripting::handleScriptReloads(atcg::Project::getActive()->getActiveScene());
            }
            else
            {
                current_operation = atcg::GuizmoOperation::ROTATE;
            }
        }
        if(event->getKeyCode() == ATCG_KEY_S)
        {
            current_operation = atcg::GuizmoOperation::SCALE;
        }
        // if(event->getKeyCode() == ATCG_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        return true;
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(in_viewport && event->getMouseButton() == ATCG_MOUSE_BUTTON_LEFT && !atcg::isOverGuizmo())
        {
            atcg::Project::getActive()->getActiveScene()->selectEntity(atcg::Utils::pickEntity(mouse_pos));
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
#endif

private:
    bool saved = false;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::ref_ptr<atcg::Graph> plane;

    atcg::GUI::SceneHierarchyPanel panel;
    atcg::GUI::PerformancePanel performance_panel;
    atcg::GUI::AssetPanel asset_panel;
    bool show_performance = false;

    float time       = 0.0f;
    bool in_viewport = false;

    glm::vec2 mouse_pos;

    atcg::GUI::RenderSettingsPanel render_settings_panel;
    bool show_render_settings     = false;
    bool show_integrator_settings = true;

#ifndef ATCG_HEADLESS
    atcg::GuizmoOperation current_operation = atcg::GuizmoOperation::TRANSLATE;
#endif

    atcg::ref_ptr<atcg::IntegratorController> integrator_controller;
    atcg::GUI::IntegratorPanel integrator_panel;
};

class NRCApp : public atcg::Application
{
public:
    NRCApp(const atcg::WindowProps& props) : atcg::Application(props) { pushLayer(new NRCLayer("Layer")); }

    ~NRCApp() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = true;
    return new NRCApp(props);
}