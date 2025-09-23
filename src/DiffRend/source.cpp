#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <algorithm>

#include <random>
#include <stb_image.h>
#include <portable-file-dialogs.h>

#include <Core/Common.h>

#include "DiffPathtracingIntegrator.h"

class DiffRendLayer : public atcg::Layer
{
public:
    void createOutputTexture(int width, int height)
    {
#ifdef ATCG_ENABLE_OPTIX
        output_tensor   = torch::zeros({height, width, 4}, atcg::TensorOptions::uint8DeviceOptions());
        output_entities = torch::zeros({height, width}, atcg::TensorOptions::int32DeviceOptions());

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
#ifdef ATCG_ENABLE_OPTIX
        pipeline = atcg::make_ref<atcg::RayTracingPipeline>(optx_context);
        sbt      = atcg::make_ref<atcg::ShaderBindingTable>();

        integrator = atcg::make_ref<atcg::DiffPathtracingIntegrator>(optx_context, atcg::Dictionary());
        integrator->setScene(atcg::Project::getActive()->getActiveScene());
        integrator->initializePipeline(pipeline, sbt);

        pipeline->createPipeline();
        sbt->createSBT();
#endif
    }

    DiffRendLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

        atcg::Project::load("../DiffRendTest/Project.json");

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        atcg::CameraIntrinsics intrinsics;
        intrinsics.setAspectRatio(aspect_ratio);
        camera_controller = atcg::make_ref<atcg::FirstPersonController>(
            atcg::make_ref<atcg::PerspectiveCamera>(atcg::CameraExtrinsics(), intrinsics));


        atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());

#ifdef ATCG_ENABLE_OPTIX
        optx_context = atcg::RaytracingContextManager::createContext();
        if(enable_pathtracing) initializePathtracer();
#endif

        createOutputTexture(atcg::Renderer::getFramebuffer()->width(), atcg::Renderer::getFramebuffer()->height());

        atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        performance_panel.registerFrameTime(delta_time);
        bool updated = camera_controller->onUpdate(delta_time);

#ifdef ATCG_ENABLE_OPTIX
        if(enable_pathtracing && updated)
        {
            integrator->reset();
        }
#endif

        atcg::Scripting::handleScriptUpdates(atcg::Project::getActive()->getActiveScene(), delta_time);

        atcg::Renderer::clear();


        atcg::Renderer::clear();

        if(enable_pathtracing)
        {
#ifdef ATCG_ENABLE_OPTIX
            atcg::Dictionary dict;
            dict.setValue("camera", camera_controller->getCamera());
            dict.setValue("output", output_tensor);
            dict.setValue("entity_ids", output_entities);
            integrator->generateRays(dict);
            output_texture->setData(output_tensor);
            output_entity_texture->setData(output_entities);

            atcg::Renderer::drawImage(output_texture, output_entity_texture);
#endif
        }
        else
        {
            atcg::Project::getActive()->getActiveScene()->draw(camera_controller->getCamera(),
                                                               atcg::Renderer::getFramebuffer());
        }


        atcg::Renderer::drawCameras(atcg::Project::getActive()->getActiveScene(), camera_controller->getCamera());
        atcg::Renderer::drawLights(atcg::Project::getActive()->getActiveScene(), camera_controller->getCamera());

        atcg::Renderer::drawCADGrid(camera_controller->getCamera());


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

            if(ImGui::Checkbox("Enable MSAA", &msaa_enabled))
            {
                auto graph = msaa_enabled ? atcg::createMSAAGraph(msaa_samples[current_msaa_selection_index])
                                          : atcg::createStandardGraph();
                atcg::Project::getActive()->getActiveScene()->setRenderGraph(graph);
            }

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
                            atcg::Project::getActive()->getActiveScene()->setRenderGraph(
                                atcg::createMSAAGraph(msaa_samples[current_msaa_selection_index]));
                        }

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if(is_selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }

    #ifdef ATCG_ENABLE_OPTIX
            if(ImGui::Checkbox("Path Tracing", &enable_pathtracing))
            {
                if(enable_pathtracing) initializePathtracer();
            }
    #endif

            ImGui::End();
        }

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
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(DiffRendLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(DiffRendLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(DiffRendLayer::onKeyPressed));
#endif
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(DiffRendLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);
        createOutputTexture(event->getWidth(), event->getHeight());
        return false;
    }

#ifndef ATCG_HEADLESS
    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == ATCG_KEY_T)
        {
            current_operation = ImGuizmo::OPERATION::TRANSLATE;
        }
        if(event->getKeyCode() == ATCG_KEY_R)
        {
            if(atcg::Input::isKeyPressed(ATCG_KEY_LEFT_CONTROL))
            {
                atcg::Scripting::handleScriptReloads(atcg::Project::getActive()->getActiveScene());
            }
            else
            {
                current_operation = ImGuizmo::OPERATION::ROTATE;
            }
        }
        if(event->getKeyCode() == ATCG_KEY_S)
        {
            current_operation = ImGuizmo::OPERATION::SCALE;
        }
        // if(event->getKeyCode() == ATCG_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        return true;
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(in_viewport && event->getMouseButton() == ATCG_MOUSE_BUTTON_LEFT && !ImGuizmo::IsOver())
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1
                                 ? atcg::Entity()
                                 : atcg::Entity((entt::entity)id, atcg::Project::getActive()->getActiveScene().get());
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
    uint32_t current_msaa_selection_index = 4;
    bool msaa_enabled                     = true;
#ifndef ATCG_HEADLESS
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
#endif

#ifdef ATCG_ENABLE_OPTIX
    atcg::ref_ptr<atcg::RaytracingContext> optx_context;
    atcg::ref_ptr<atcg::RayTracingPipeline> pipeline;
    atcg::ref_ptr<atcg::ShaderBindingTable> sbt;
    atcg::ref_ptr<atcg::DiffPathtracingIntegrator> integrator;
#endif

    torch::Tensor output_tensor;
    atcg::ref_ptr<atcg::Texture2D> output_texture;

    torch::Tensor output_entities;
    atcg::ref_ptr<atcg::Texture2D> output_entity_texture;

    uint32_t last_revision = 0;
};

class DiffRend : public atcg::Application
{
public:
    DiffRend(const atcg::WindowProps& props) : atcg::Application(props) { pushLayer(new DiffRendLayer("Layer")); }

    ~DiffRend() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = true;
    return new DiffRend(props);
}