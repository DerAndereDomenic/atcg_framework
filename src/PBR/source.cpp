#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>
#include <portable-file-dialogs.h>

#include <Core/Common.h>

class PBRLayer : public atcg::Layer
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

        integrator = atcg::make_ref<atcg::VolPathtracingIntegrator>(optx_context, dict);
#endif
    }

    PBRLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);

        auto skybox         = atcg::IO::imread((atcg::resource_directory() / "pbr/skybox.hdr").string());
        auto skybox_texture = atcg::Texture2D::create(skybox);
        atcg::AssetManager::registerAsset(skybox_texture, "skybox");

        auto scene = atcg::IO::read_scene((atcg::resource_directory() / "test_scene.obj").string());
        atcg::Project::getActive()->setActiveScene(scene);
        atcg::Project::getActive()->getActiveScene()->setSkybox(skybox_texture);

        {
            auto sphere  = atcg::Project::getActive()->getActiveScene()->getEntitiesByName("Icosphere").front();
            auto& script = sphere.addComponent<atcg::ScriptComponent>(atcg::make_ref<atcg::PythonScript>("./src/PBR/"
                                                                                                         "bounce.py"));
            script.script()->init();
            auto behavior = script.behavior(atcg::Project::getActive()->getActiveScene(), sphere);

            if(behavior)
            {
                behavior->onAttach();
            }
        }

        if(atcg::VR::isVRAvailable())
        {
            float vr_aspect = (float)atcg::VR::width() / (float)atcg::VR::height();

            atcg::CameraIntrinsics instrinsics_left(atcg::VR::getProjection(atcg::VRSystem::Eye::LEFT));
            atcg::CameraIntrinsics instrinsics_right(atcg::VR::getProjection(atcg::VRSystem::Eye::RIGHT));

            atcg::CameraExtrinsics extrinsics_left(glm::inverse(atcg::VR::getInverseView(atcg::VRSystem::Eye::LEFT)));
            atcg::CameraExtrinsics extrinsics_right(glm::inverse(atcg::VR::getInverseView(atcg::VRSystem::Eye::RIGHT)));

            camera_controller = atcg::make_ref<atcg::VRController>(
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics_left, instrinsics_left),
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics_right, instrinsics_right));
            atcg::VR::initControllerMeshes(atcg::Project::getActive()->getActiveScene());
        }
        else
        {
            const auto& window = atcg::Application::get()->getWindow();
            float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
            atcg::CameraIntrinsics intrinsics;
            intrinsics.setAspectRatio(aspect_ratio);
            atcg::CameraExtrinsics extrinsics;
            extrinsics.setPosition(glm::vec3(0, 1, 3));
            extrinsics.setTarget(glm::vec3(0, 1, 0));
            camera_controller = atcg::make_ref<atcg::FirstPersonController>(
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics));
        }

        atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());

        // Instance Buffer Test
        {
            auto sphere = atcg::IO::read_mesh((atcg::resource_directory() / "armadillo.obj").string());
            atcg::AssetManager::registerAsset(sphere, "sphere");
            int n_instances = 100;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> posDist(-5.0f, 5.0f);
            std::uniform_real_distribution<float> colorDist(0.0f, 1.0f);

            std::vector<glm::mat4> transforms;
            std::vector<glm::vec4> colors;

            for(int i = 0; i < n_instances; ++i)
            {
                glm::vec3 pos   = {posDist(gen), posDist(gen) + 10.0f, posDist(gen)};
                glm::vec3 color = {colorDist(gen), colorDist(gen), colorDist(gen)};
                glm::vec3 scale = glm::vec3(colorDist(gen) * 0.03f);
                glm::vec3 axis  = {colorDist(gen), colorDist(gen), colorDist(gen)};
                axis            = glm::normalize(2.0f * axis - 1.0f);
                float angle     = atcg::Constants::two_pi<float>() * colorDist(gen);
                transforms.push_back(glm::translate(pos) * glm::rotate(angle, axis) * glm::scale(scale));
                colors.push_back(glm::vec4(color, 1));
            }

            atcg::ref_ptr<atcg::VertexBuffer> vbo_transforms =
                atcg::make_ref<atcg::VertexBuffer>(transforms.data(), transforms.size() * sizeof(glm::mat4));
            vbo_transforms->setLayout({{atcg::ShaderDataType::Mat4, "transform"}});
            atcg::ref_ptr<atcg::VertexBuffer> vbo_colors =
                atcg::make_ref<atcg::VertexBuffer>(colors.data(), colors.size() * sizeof(glm::vec4));
            vbo_colors->setLayout({{atcg::ShaderDataType::Float4, "color"}});

            auto entity = atcg::Project::getActive()->getActiveScene()->createEntity("Instances");
            entity.addComponent<atcg::TransformComponent>();
            entity.addComponent<atcg::GeometryComponent>(sphere);
            auto& instances = entity.addComponent<atcg::InstanceRenderComponent>();
            instances.addInstanceBuffer(vbo_transforms);
            instances.addInstanceBuffer(vbo_colors);
        }

#ifdef ATCG_CUDA_BACKEND
        optx_context = atcg::RaytracingContextManager::createContext();
#endif

        createOutputTexture(atcg::Renderer::getFramebuffer()->width(), atcg::Renderer::getFramebuffer()->height());

        atcg::CompileData compile_data;
        compile_data.num_samples = msaa_samples[current_msaa_selection_index];
        auto render_graph        = atcg::createRenderGraph(compile_data);

        atcg::SceneRenderer::setRenderGraph(render_graph);
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

        if(atcg::VR::isVRAvailable())
        {
            atcg::ref_ptr<atcg::VRController> controller =
                std::dynamic_pointer_cast<atcg::VRController>(camera_controller);

            if(controller->inMovement())
            {
                atcg::VR::setMovementLine(controller->getControllerPosition(), controller->getControllerIntersection());
            }

            auto [t_left, t_right] = atcg::VR::getRenderTargets();

            // t_left->use();
            //  atcg::Renderer::setViewport(0, 0, atcg::VR::width(), atcg::VR::height());

            // atcg::Renderer::clear();

            // TODO
            // atcg::Project::getActive()->getActiveScene()->draw(controller->getCameraLeft(), t_left);

            atcg::Renderer::drawCADGrid(controller->getCameraLeft());

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraLeft());
            }

            // t_right->use();

            // atcg::Renderer::clear();

            // TODO
            // atcg::Project::getActive()->getActiveScene()->draw(controller->getCameraRight(), t_right);

            atcg::Renderer::drawCADGrid(controller->getCameraRight());

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraRight());
            }

            // atcg::Renderer::useScreenBuffer();
            // atcg::Renderer::setDefaultViewport();

            atcg::VR::renderToScreen();
        }
        else
        {
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
                            atcg::CompileData compile_data;
                            compile_data.num_samples = msaa_samples[current_msaa_selection_index];
                            auto render_graph        = atcg::createRenderGraph(compile_data);
                            atcg::SceneRenderer::setRenderGraph(render_graph);
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
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onKeyPressed));
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onFileDropped));
#endif
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onViewportResized));
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

    bool onFileDropped(atcg::FileDroppedEvent* event)
    {
        for(int i = 0; i < event->getCount(); ++i)
        {
            std::filesystem::path filepath = event->getPath(i);

            auto file_ending = filepath.extension().string();

            if(file_ending == ".obj")
            {
                auto graph = atcg::IO::read_mesh(filepath.string());
                atcg::AssetManager::registerAsset(graph, filepath.stem().string());
            }
            else if(file_ending == ".png" || file_ending == ".jpg" || file_ending == ".jpeg" || file_ending == ".hdr")
            {
                auto img     = atcg::IO::imread(filepath.string());
                auto texture = atcg::Texture2D::create(img);
                atcg::AssetManager::registerAsset(texture, filepath.stem().string());
            }
        }

        return true;
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

    bool enable_pathtracing = false;

    uint32_t msaa_samples[6]              = {1, 2, 4, 8, 16, 32};
    const char* msaa_samples_str[6]       = {"1", "2", "4", "8", "16", "32"};
    uint32_t current_msaa_selection_index = 0;
    bool msaa_enabled                     = true;
#ifndef ATCG_HEADLESS
    atcg::GuizmoOperation current_operation = atcg::GuizmoOperation::TRANSLATE;
#endif

#ifdef ATCG_CUDA_BACKEND
    atcg::ref_ptr<atcg::RaytracingContext> optx_context;
    atcg::ref_ptr<atcg::VolPathtracingIntegrator> integrator;
#endif

    atcg::ref_ptr<atcg::Texture2D> output_texture;
    atcg::ref_ptr<atcg::Texture2D> output_entity_texture;

    uint32_t last_revision = 0;
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