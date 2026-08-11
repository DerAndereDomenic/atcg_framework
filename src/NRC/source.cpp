#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>
#include <portable-file-dialogs.h>

#include <Core/Optix.h>

#include "NRCIntegrator.h"

class NRCLayer : public atcg::Layer
{
public:
    void createOutputTexture(int width, int height)
    {
#ifdef ATCG_CUDA_BACKEND
        atcg::TextureSpecification spec;
        spec.width     = width;
        spec.height    = height;
        spec.format    = atcg::TextureFormat::RGBA;
        output_texture = atcg::Texture2D::create(spec);

        atcg::TextureSpecification spec_int;
        spec_int.width        = width;
        spec_int.height       = height;
        spec_int.format       = atcg::TextureFormat::RINT;
        output_entity_texture = atcg::Texture2D::create(spec_int);
#endif
    }

    void initializePathtracer()
    {
#ifdef ATCG_CUDA_BACKEND
        atcg::Dictionary dict;
        dict.setValue<atcg::ref_ptr<atcg::Scene>>("scene", atcg::Project::getActive()->getActiveScene());
        dict.setValue<uint32_t>("width", atcg::Renderer::getFramebuffer()->width());
        dict.setValue<uint32_t>("height", atcg::Renderer::getFramebuffer()->height());

        integrator = atcg::make_ref<atcg::NRCIntegrator>(optx_context, dict);
#endif
    }

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

#ifdef ATCG_CUDA_BACKEND
        optx_context = atcg::RaytracingContextManager::createContext();
        if(enable_pathtracing) initializePathtracer();
#endif

        createOutputTexture(atcg::Renderer::getFramebuffer()->width(), atcg::Renderer::getFramebuffer()->height());

        atcg::SceneRenderer::setNumberMSAASamples(msaa_samples[current_msaa_selection_index]);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        performance_panel.registerFrameTime(delta_time);
        bool updated = camera_controller->onUpdate(delta_time);

#ifdef ATCG_CUDA_BACKEND
        if(enable_pathtracing && updated)
        {
            integrator->reset();
        }
#endif

        atcg::Scripting::handleScriptUpdates(atcg::Project::getActive()->getActiveScene(), delta_time);


        // atcg::Renderer::clear();

        if(enable_pathtracing)
        {
#ifdef ATCG_CUDA_BACKEND
            atcg::Dictionary dict;
            integrator->generateRays(dict);
            torch::Tensor output_tensor   = dict.getValue<torch::Tensor>("output");
            torch::Tensor output_entities = dict.getValue<torch::Tensor>("entity_ids");
            output_texture->setData(output_tensor);
            output_entity_texture->setData(output_entities);

            atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());
            atcg::GraphicsCommand::clear();
            atcg::Renderer::drawImage(output_texture, output_entity_texture);
            atcg::GraphicsCommand::endRenderPass();
#endif
        }
        else
        {
            atcg::SceneRenderer::render(atcg::Project::getActive()->getActiveScene(),
                                        camera_controller->getCamera(),
                                        atcg::Renderer::getFramebuffer());

            atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());

            atcg::Renderer::drawCADGrid(camera_controller->getCamera());
            atcg::GraphicsCommand::endRenderPass();
        }


        uint32_t current_revision = atcg::RevisionStack::numUndos();
        if(current_revision != last_revision)
        {
            last_revision = current_revision;
            if(enable_pathtracing) initializePathtracer();
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
                        initializePathtracer();
                    }
                    atcg::RevisionStack::clearChache();

                    hovered_entity = atcg::Entity();
                    saved          = true;
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
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);
            if(ImGui::Checkbox("VSync", &vsync))
            {
                atcg::Application::get()->getWindow()->toggleVSync(vsync);
            }

            const char* combo_preview_value = msaa_samples_str[current_msaa_selection_index];

            if(msaa_enabled)
            {
                if(ImGui::BeginCombo("MSAA Samples", combo_preview_value))
                {
                    for(int n = 0; n < IM_ARRAYSIZE(msaa_samples); n++)
                    {
                        const bool is_selected = (current_msaa_selection_index == n);
                        if(ImGui::Selectable(msaa_samples_str[n], is_selected))
                        {
                            current_msaa_selection_index = n;
                            atcg::SceneRenderer::setNumberMSAASamples(msaa_samples[current_msaa_selection_index]);
                        }

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if(is_selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }

    #ifdef ATCG_CUDA_BACKEND
            if(ImGui::Checkbox("Path Tracing", &enable_pathtracing))
            {
                if(enable_pathtracing) initializePathtracer();
            }
    #endif

            ImGui::End();
        }

    #ifdef ATCG_CUDA_BACKEND
        if(enable_pathtracing) integrator->onImGuiRender();
    #endif

        performance_panel.renderPanel(show_performance);
        panel.renderPanel(atcg::Project::getActive()->getActiveScene());
        hovered_entity = panel.getSelectedEntity();

        asset_panel.renderPanel();

        atcg::drawGuizmo(atcg::Project::getActive()->getActiveScene(),
                         hovered_entity,
                         current_operation,
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
        createOutputTexture(event->getWidth(), event->getHeight());
        if(enable_pathtracing) initializePathtracer();
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
            hovered_entity = atcg::Utils::pickEntity(mouse_pos);
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
#endif

private:
    bool saved = false;

    atcg::Entity hovered_entity;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::ref_ptr<atcg::Graph> plane;

    atcg::GUI::SceneHierarchyPanel panel;
    atcg::GUI::PerformancePanel performance_panel;
    atcg::GUI::AssetPanel asset_panel;
    bool show_performance = false;

    float time       = 0.0f;
    bool in_viewport = false;

    glm::vec2 mouse_pos;

    bool show_render_settings = false;
    bool vsync                = true;

    bool enable_pathtracing = true;

    uint32_t msaa_samples[6]              = {1, 2, 4, 8, 16, 32};
    const char* msaa_samples_str[6]       = {"1", "2", "4", "8", "16", "32"};
    uint32_t current_msaa_selection_index = 0;
    bool msaa_enabled                     = true;
#ifndef ATCG_HEADLESS
    atcg::GuizmoOperation current_operation = atcg::GuizmoOperation::TRANSLATE;
#endif

#ifdef ATCG_CUDA_BACKEND
    atcg::ref_ptr<atcg::RaytracingContext> optx_context;
    atcg::ref_ptr<atcg::Integrator> integrator;
#endif

    atcg::ref_ptr<atcg::Texture2D> output_texture;
    atcg::ref_ptr<atcg::Texture2D> output_entity_texture;

    uint32_t last_revision = 0;
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