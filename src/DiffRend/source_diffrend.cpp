#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <algorithm>

#include <random>
#include <stb_image.h>
#include <portable-file-dialogs.h>

#include <Core/Common.h>
#include <torch/optim.h>

#include "AttachedDiffPathtracingIntegrator.h"
#include "DiffPathtracingIntegrator.h"
#include "FiniteDiffPathIntegrator.h"
#include "VolDiffPathtracingIntegrator.h"

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

class DiffRendLayer : public atcg::Layer
{
public:
    void createOutputTexture(int width_, int height_)
    {
        int width  = width_ / 4;
        int height = height_ / 4;
#ifdef ATCG_ENABLE_OPTIX

        atcg::TextureSpecification spec;
        spec.width     = width;
        spec.height    = height;
        spec.format    = atcg::TextureFormat::RGBA;
        output_texture = atcg::Texture2D::create(spec);
#endif
    }

    void initializePathtracer()
    {
#ifdef ATCG_ENABLE_OPTIX
        atcg::Dictionary dict;
        dict.setValue<atcg::ref_ptr<atcg::Scene>>("scene", atcg::Project::getActive()->getActiveScene());
        dict.setValue<uint32_t>("width", atcg::Renderer::getFramebuffer()->width() / 4);
        dict.setValue<uint32_t>("height", atcg::Renderer::getFramebuffer()->height() / 4);

        if(current_integrator_index == 0)
        {
            integrator = atcg::make_ref<atcg::AttachedDiffPathtracingIntegrator>(optx_context, dict);
        }
        else if(current_integrator_index == 1)
        {
            integrator = atcg::make_ref<atcg::DiffPathtracingIntegrator>(optx_context, dict);
        }
        else if(current_integrator_index == 2)
        {
            integrator = atcg::make_ref<atcg::FiniteDiffPathtracingIntegrator>(optx_context, dict);
        }
        else if(current_integrator_index == 3)
        {
            integrator = atcg::make_ref<atcg::VolDiffPathtracingIntegrator>(optx_context, dict);
        }
#endif
    }

    DiffRendLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);

        atcg::Project::load("../DiffRendTest_old/Project.json");

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
            frame_counter = 0;
        }
#endif

        atcg::Scripting::handleScriptUpdates(atcg::Project::getActive()->getActiveScene(), delta_time);

        if(enable_pathtracing)
        {
#ifdef ATCG_ENABLE_OPTIX

            if(optimize)
            {
                optimizer->zero_grad(false);
                // integrator->zeroGrad();
                uint32_t num_samples = 128;

                torch::Tensor result = torch::zeros({output_texture->height(), output_texture->width(), 3},
                                                    atcg::TensorOptions::floatDeviceOptions());

                torch::Tensor fake_result = torch::zeros({output_texture->height(), output_texture->width(), 3},
                                                         atcg::TensorOptions::floatDeviceOptions());

                for(int i = 0; i < num_samples; ++i)
                {
                    atcg::Dictionary dict;
                    dict.setValue("camera", camera_controller->getCamera());
                    dict.setValue("width", output_texture->width());
                    dict.setValue("height", output_texture->height());
                    dict.setValue("rng_index", iteration_count * num_samples + i);

                    result = result + integrator->sample(dict) / (float)num_samples;

                    {
                        torch::NoGradGuard no_grad;
                        dict.setValue<uint32_t>("rng_index", 1e6 + iteration_count * num_samples + i);

                        fake_result = fake_result + integrator->sample(dict) / (float)num_samples;
                    }
                }

                torch::Tensor result_injected = result + (fake_result - result).detach();

                auto difference = (result_injected - target) * (result_injected - target);
                auto L          = torch::sum(torch::abs(difference));

                L.backward();
                optimizer->step();

                ATCG_TRACE("Iteration {}: Loss = {}", iteration_count, L.item<float>());
                time_collection.addSample((float)iteration_count);
                loss_collection.addSample(L.item<float>());

                ++iteration_count;

                {
                    torch::NoGradGuard no_grad;

                    integrator->clampParameters();
                    difference_texture->setData(torch::abs(difference));
                    result_texture->setData(result);

                    torch::Tensor tonemapped = torch::pow(1.0f - torch::exp(-result), 1.0 / 2.4f);
                    tonemapped.clamp_(0.0f, 1.0f);

                    torch::Tensor output_img = torch::full({output_texture->height(), output_texture->width(), 4},
                                                           255,
                                                           atcg::TensorOptions::uint8DeviceOptions());
                    output_img.index_put_(
                        {torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                        (tonemapped * 255.0f).to(torch::kUInt8));
                    output_texture->setData(output_img);
                }
            }
            else
            {
                torch::NoGradGuard no_grad;

                atcg::Dictionary dict;
                dict.setValue("camera", camera_controller->getCamera());
                dict.setValue("width", output_texture->width());
                dict.setValue("height", output_texture->height());
                dict.setValue("debug", debug);
                dict.setValue("rng_index", frame_counter);
                integrator->generateRays(dict);
                auto output = dict.getValue<torch::Tensor>("output_img");

                if(frame_counter > 0)
                {
                    float alpha        = 1.0f / (float)(frame_counter + 1);
                    accumulated_output = accumulated_output * (1.0f - alpha) + output * alpha;
                }
                else
                {
                    accumulated_output = output;
                }

                torch::Tensor tonemapped = torch::pow(1.0f - torch::exp(-accumulated_output), 1.0 / 2.4f);
                tonemapped.clamp_(0.0f, 1.0f);

                torch::Tensor output_img = torch::full({output_texture->height(), output_texture->width(), 4},
                                                       255,
                                                       atcg::TensorOptions::uint8DeviceOptions());
                output_img.index_put_(
                    {torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                    (tonemapped * 255.0f).to(torch::kUInt8));
                output_texture->setData(output_img);

                ++frame_counter;

                if(frame_counter == 1024)
                {
                    target = accumulated_output.clone();

                    target_texture     = atcg::Texture2D::create(target);
                    difference_texture = atcg::Texture2D::create(torch::zeros_like(target));
                    result_texture     = atcg::Texture2D::create(torch::zeros_like(target));
                }
            }

            atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());
            atcg::GraphicsCommand::clear();
            atcg::Renderer::drawImage(output_texture);
            atcg::GraphicsCommand::endRenderPass();
#endif
        }
        else
        {
            atcg::Project::getActive()->getActiveScene()->draw(camera_controller->getCamera(),
                                                               atcg::Renderer::getFramebuffer());
        }

        atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());

        atcg::Renderer::drawCADGrid(camera_controller->getCamera());
        atcg::GraphicsCommand::endRenderPass();


        uint32_t current_revision = atcg::RevisionStack::numUndos();
        if(current_revision != last_revision)
        {
            last_revision = current_revision;
            frame_counter = 0;
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
                        frame_counter = 0;
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

            ImGui::Checkbox("Enable Debug", &debug);

            ImGui::End();
        }

        ImGui::Begin("Optimization");

        // Dropdown to select attached, finite or detached integrator
        if(ImGui::BeginCombo("Integrator", integrator_labels[current_integrator_index]))
        {
            for(int n = 0; n < IM_ARRAYSIZE(integrator_labels); n++)
            {
                const bool is_selected = (current_integrator_index == n);
                if(ImGui::Selectable(integrator_labels[n], is_selected))
                {
                    current_integrator_index = n;
                    // if(current_integrator_index == 0)
                    // {
                    //     integrator =
                    //         atcg::make_ref<atcg::AttachedDiffPathtracingIntegrator>(*optx_context,
                    //         atcg::Dictionary());
                    // }
                    // else if(current_integrator_index == 1)
                    // {
                    //     integrator = atcg::make_ref<atcg::DiffPathtracingIntegrator>(*optx_context,
                    //     atcg::Dictionary());
                    // }
                    // else
                    // {
                    //     integrator =
                    //         atcg::make_ref<atcg::FiniteDiffPathtracingIntegrator>(*optx_context, atcg::Dictionary());
                    // }

                    initializePathtracer();
                }

                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if(is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if(ImGui::Button("Register target"))
        {
            target = accumulated_output.clone();

            target_texture     = atcg::Texture2D::create(target);
            difference_texture = atcg::Texture2D::create(torch::zeros_like(target));
            result_texture     = atcg::Texture2D::create(torch::zeros_like(target));
        }

        if(ImGui::Button("Toggle Optimization"))
        {
            frame_counter = 0;
            optimize      = !optimize;
            if(optimize)
            {
                // integrator->markOptimizable();
                ATCG_DEBUG("Optimizing {} parameters", integrator->getParameters().size());
                optimizer =
                    atcg::make_ref<torch::optim::Adam>(integrator->getParameters(), torch::optim::AdamOptions(0.01));
                iteration_count = 0;
                time_collection.resetStatistics();
                loss_collection.resetStatistics();
            }
        }

        // ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 100);
        if(ImPlot::BeginPlot("Loss"))
        {
            ImPlot::SetupAxes("Iteration", "Loss", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("Loss",
                             time_collection.get(),
                             loss_collection.get(),
                             loss_collection.count(),
                             0,
                             loss_collection.index(),
                             sizeof(float));
            ImPlot::EndPlot();
        }


        if(integrator)
        {
            integrator->onImGuiRender();
        }

        ImGui::Begin("Target Texture");
        if(target_texture)
        {
            ImGui::Image((ImTextureID)target_texture->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        }
        ImGui::End();

        ImGui::Begin("Result Texture");
        if(result_texture)
        {
            ImGui::Image((ImTextureID)result_texture->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        }
        ImGui::End();

        ImGui::Begin("Difference Texture");
        if(difference_texture)
        {
            ImGui::Image((ImTextureID)difference_texture->getID(), ImVec2(512, 512), ImVec2 {0, 1}, ImVec2 {1, 0});
        }
        ImGui::End();

        ImGui::End();

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
        frame_counter = 0;
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

    bool show_render_settings = true;
    bool vsync                = true;
    bool debug                = false;

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
    atcg::ref_ptr<atcg::DifferentiableIntegrator> integrator;
    const char* integrator_labels[4]  = {"Attached", "Detached", "Finite Difference", "VolDetached"};
    uint32_t current_integrator_index = 1;
    torch::Tensor target;
    torch::Tensor accumulated_output;
    bool optimize          = false;
    int iteration_count    = 0;
    uint32_t frame_counter = 0;
    atcg::ref_ptr<torch::optim::Adam> optimizer;
    atcg::ref_ptr<atcg::Texture2D> target_texture;
    atcg::ref_ptr<atcg::Texture2D> difference_texture;
    atcg::ref_ptr<atcg::Texture2D> result_texture;
#endif

    atcg::ref_ptr<atcg::Texture2D> output_texture;

    uint32_t last_revision = 0;

    atcg::CyclicCollection<float> time_collection = atcg::CyclicCollection<float>("Time Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> loss_collection = atcg::CyclicCollection<float>("Loss Collection", 35 * 60 / 5);
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
    props.width  = 3000;
    props.height = 1800;
    props.vsync  = true;
    return new DiffRend(props);
}