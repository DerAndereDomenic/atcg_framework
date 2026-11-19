#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>
#include <portable-file-dialogs.h>

#include <Core/Common.h>

class EditorLayer : public atcg::Layer
{
public:
    EditorLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);

        auto scene = atcg::make_ref<atcg::Scene>();
        atcg::Project::getActive()->setActiveScene(scene);

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        atcg::CameraIntrinsics intrinsics;
        intrinsics.setAspectRatio(aspect_ratio);
        atcg::CameraExtrinsics extrinsics;
        extrinsics.setPosition(glm::vec3(-3, 3, 3));
        extrinsics.setTarget(glm::vec3(0, 1, 0));
        camera_controller = atcg::make_ref<atcg::FirstPersonController>(
            atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics));


        atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());

#ifdef ATCG_CUDA_BACKEND
        integrator_controller = atcg::make_ref<atcg::IntegratorController>();
        integrator_panel      = atcg::GUI::IntegratorPanel(integrator_controller);
#endif

        // Create default scene
        {
            auto cube_entity = scene->createEntity("Cube");
            cube_entity.addComponent<atcg::TransformComponent>();
            cube_entity.addComponent<atcg::GeometryComponent>(atcg::AssetManager::getCubeMesh());
            cube_entity.addComponent<atcg::MeshRenderComponent>();
        }

        {
            auto light_entity = scene->createEntity("Light");
            auto& transform   = light_entity.addComponent<atcg::TransformComponent>();
            transform.setPosition(glm::vec3(-2, 2.4, 1));
            auto& light     = light_entity.addComponent<atcg::PointLightComponent>();
            light.intensity = 10.0f;
        }
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        performance_panel.registerFrameTime(delta_time);
        camera_controller->onUpdate(delta_time);

        atcg::Scripting::handleScriptUpdates(atcg::Project::getActive()->getActiveScene(), delta_time);

        auto scene = atcg::Project::getActive()->getActiveScene();

#ifdef ATCG_CUDA_BACKEND
        // The controller rebuilds or resets the integrator itself if the scene, camera or viewport changed
        if(integrator_controller->render(scene, atcg::Renderer::getFramebuffer())) return;
#endif

        atcg::SceneRenderer::render(scene, camera_controller->getCamera(), atcg::Renderer::getFramebuffer());

        atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());
        atcg::Renderer::drawCADGrid(camera_controller->getCamera());
        atcg::GraphicsCommand::endRenderPass();
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

                    // The integrator controller picks up the loaded integrator settings on its own
                    if(atcg::Project::getActive()->getActiveScene())
                    {
                        atcg::Project::getActive()->getActiveScene()->setCamera(camera_controller->getCamera());
                    }
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
    #ifdef ATCG_CUDA_BACKEND
            ImGui::MenuItem("Show Integrator Settings", nullptr, &show_integrator_settings);
    #endif
            ImGui::EndMenu();
        }

        if(ImGui::BeginMenu("Editor"))
        {
            ImGui::MenuItem("Show Editor Settings", nullptr, &show_editor_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_editor_settings)
        {
            ImGui::Begin("Editor Settings", &show_editor_settings);
            ImGui::Checkbox("Create entities on import", &create_entities_on_import);
            ImGui::End();
        }

        render_settings_panel.renderPanel(show_render_settings);
    #ifdef ATCG_CUDA_BACKEND
        integrator_panel.renderPanel(show_integrator_settings);
    #endif

        performance_panel.renderPanel(show_performance);
        // The active entity is only outlined if no asset was selected after it
        panel.setEntitySelectionActive(!asset_panel.isAssetSelectionActive());
        panel.renderPanel(atcg::Project::getActive()->getActiveScene());
        plugin_panel.renderPanel();

        asset_panel.renderPanel();

        atcg::drawGuizmo(atcg::Project::getActive()->getActiveScene(),
                         current_operation,
                         current_mode,
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
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(EditorLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(EditorLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(EditorLayer::onKeyPressed));
        dispatcher.dispatch<atcg::FileDroppedEvent>(ATCG_BIND_EVENT_FN(EditorLayer::onFileDropped));
        dispatcher.dispatch<atcg::AssetDroppedEvent>(ATCG_BIND_EVENT_FN(EditorLayer::onAssetDropped));
#endif
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(EditorLayer::onViewportResized));
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
        if(event->getKeyCode() == ATCG_KEY_Z)
        {
            current_operation = atcg::GuizmoOperation::SCALE;
        }

        if(event->getKeyCode() == ATCG_KEY_L)
        {
            current_mode =
                (current_mode == atcg::GuizmoMode::LOCAL) ? atcg::GuizmoMode::WORLD : atcg::GuizmoMode::LOCAL;
        }

        // Copy & paste. Ctrl shortcuts are passed on by the ImGui layer independent of the focused window.
        bool ctrl =
            atcg::Input::isKeyPressed(ATCG_KEY_LEFT_CONTROL) || atcg::Input::isKeyPressed(ATCG_KEY_RIGHT_CONTROL);
        if(ctrl && event->getKeyCode() == ATCG_KEY_C && !event->IsRepeat())
        {
            copySelection();
        }
        if(ctrl && event->getKeyCode() == ATCG_KEY_V && !event->IsRepeat())
        {
            pasteClipboard();
        }

        if(event->getKeyCode() == ATCG_KEY_DELETE && !atcg::RevisionStack::isRecording())
        {
            // Delete all selected entities in one revision. Removing an entity also removes it from the selection.
            auto scene                          = atcg::Project::getActive()->getActiveScene();
            std::vector<atcg::Entity> to_delete = scene->getSelectedEntities();
            if(!to_delete.empty())
            {
                atcg::RevisionStack::startRecording<atcg::MultiEntityRevision<atcg::EntityRemovedRevision>>(scene,
                                                                                                            to_delete);
                for(auto entity: to_delete)
                {
                    scene->removeEntity(entity);
                }
                atcg::RevisionStack::endRecording();
            }
        }

        // if(event->getKeyCode() == ATCG_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        return true;
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(in_viewport && event->getMouseButton() == ATCG_MOUSE_BUTTON_LEFT && !atcg::isOverGuizmo())
        {
            auto scene          = atcg::Project::getActive()->getActiveScene();
            atcg::Entity picked = atcg::Utils::pickEntity(mouse_pos);

            if(atcg::Input::isKeyPressed(ATCG_KEY_LEFT_CONTROL) || atcg::Input::isKeyPressed(ATCG_KEY_RIGHT_CONTROL))
            {
                // Ctrl + Click adds or removes the entity. Clicking into empty space keeps the selection.
                if(picked) scene->toggleEntitySelection(picked);
            }
            else
            {
                // Clears the selection if no entity was hit
                scene->selectEntity(picked);
            }
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

    /**
     * @brief Copy (Ctrl+C). Copies the most recent selection: the selected asset, otherwise the selected entities.
     */
    void copySelection()
    {
        if(asset_panel.isAssetSelectionActive())
        {
            atcg::Clipboard::copyAsset(asset_panel.getSelectedAsset());
            return;
        }

        auto scene = atcg::Project::getActive()->getActiveScene();
        if(scene) atcg::Clipboard::copyEntities(scene->getSelectedEntities());
    }

    /**
     * @brief Paste (Ctrl+V). Entities are pasted into the active scene, assets into the asset manager.
     */
    void pasteClipboard()
    {
        auto scene = atcg::Project::getActive()->getActiveScene();
        if(atcg::Clipboard::hasEntities() && scene)
        {
            atcg::Clipboard::pasteEntities(scene);
        }
        else if(atcg::Clipboard::hasAsset())
        {
            atcg::AssetHandle pasted = atcg::Clipboard::pasteAsset();
            if(pasted != 0) asset_panel.selectAsset(pasted);
        }
    }

    bool onAssetDropped(atcg::AssetDroppedEvent* event)
    {
        // Apply the asset to the entity under the mouse (or its selection), or create a new entity from a mesh
        atcg::Utils::handleViewportAssetDrop(atcg::Project::getActive()->getActiveScene(),
                                             event->getHandle(),
                                             glm::vec2(event->getX(), event->getY()));
        return true;
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

                if(create_entities_on_import)
                {
                    auto entity = atcg::Project::getActive()->getActiveScene()->createEntity(filepath.stem().string());
                    entity.addComponent<atcg::TransformComponent>();
                    entity.addComponent<atcg::GeometryComponent>(graph);
                    auto& renderer     = entity.addComponent<atcg::MeshRenderComponent>();
                    renderer.cull_mode = atcg::CullMode::ATCG_NO_CULLING;
                }
            }
            else if(file_ending == ".png" || file_ending == ".jpg" || file_ending == ".jpeg" || file_ending == ".hdr")
            {
                auto img     = atcg::IO::imread(filepath.string());
                auto texture = atcg::Texture2D::create(img);
                atcg::AssetManager::registerAsset(texture, filepath.stem().string());
            }
            else if(file_ending == ".dll" || file_ending == ".so")
            {
                atcg::PluginManager::loadPlugin(filepath);
            }
        }

        return true;
    }
#endif

private:
    bool saved = false;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::ref_ptr<atcg::Graph> plane;

    atcg::GUI::SceneHierarchyPanel panel;
    atcg::GUI::PerformancePanel performance_panel;
    atcg::GUI::AssetPanel asset_panel;
    atcg::GUI::PluginPanel plugin_panel;
    bool show_performance = false;

    float time       = 0.0f;
    bool in_viewport = false;

    glm::vec2 mouse_pos;

    atcg::GUI::RenderSettingsPanel render_settings_panel;
    bool show_render_settings      = false;
    bool show_integrator_settings  = true;
    bool show_editor_settings      = false;
    bool create_entities_on_import = true;

#ifndef ATCG_HEADLESS
    atcg::GuizmoOperation current_operation = atcg::GuizmoOperation::TRANSLATE;
    atcg::GuizmoMode current_mode           = atcg::GuizmoMode::LOCAL;
#endif

#ifdef ATCG_CUDA_BACKEND
    atcg::ref_ptr<atcg::IntegratorController> integrator_controller;
    atcg::GUI::IntegratorPanel integrator_panel;
#endif
};

class Editor : public atcg::Application
{
public:
    Editor(const atcg::WindowProps& props) : atcg::Application(props) { pushLayer(new EditorLayer("Layer")); }

    ~Editor() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = true;
    return new Editor(props);
}