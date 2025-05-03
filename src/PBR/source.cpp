#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <algorithm>

#include <random>
#include <stb_image.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <Core/Common.h>

class PBRLayer : public atcg::Layer
{
public:
    void createOutputTexture(int width, int height)
    {
        output_tensor = torch::zeros({height, width, 4}, atcg::TensorOptions::uint8DeviceOptions());
    }

    void initializePathtracer()
    {
        pipeline = atcg::make_ref<atcg::RayTracingPipeline>(optx_context);
        sbt      = atcg::make_ref<atcg::ShaderBindingTable>(optx_context);

        integrator = atcg::make_ref<atcg::PathtracingIntegrator>(optx_context);
        integrator->setScene(scene);
        integrator->initializePipeline(pipeline, sbt);

        pipeline->createPipeline();
        sbt->createSBT();
    }

    PBRLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

        auto skybox = atcg::IO::imread((atcg::resource_directory() / "pbr/skybox.hdr").string());
        ATCG_DEBUG("{0} {1} {2}", skybox->width(), skybox->height(), skybox->channels());
        atcg::Renderer::setSkybox(skybox);

        scene = atcg::IO::read_scene((atcg::resource_directory() / "test_scene.obj").string());

        {
            auto sphere  = scene->getEntitiesByName("Icosphere").front();
            auto& script = sphere.addComponent<atcg::ScriptComponent>(atcg::make_ref<atcg::PythonScript>("./src/PBR/"
                                                                                                         "bounce.py"));
            script.script->init(scene, sphere);
            script.script->onAttach();
        }

        panel = atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>(scene);

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
            atcg::VR::initControllerMeshes(scene);
        }
        else
        {
            const auto& window = atcg::Application::get()->getWindow();
            float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
            atcg::CameraIntrinsics intrinsics;
            intrinsics.setAspectRatio(aspect_ratio);
            camera_controller = atcg::make_ref<atcg::FirstPersonController>(
                atcg::make_ref<atcg::PerspectiveCamera>(atcg::CameraExtrinsics(), intrinsics));
        }

        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        CUcontext cuCtx                   = 0;

        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optx_context));

        initializePathtracer();

        createOutputTexture(atcg::Renderer::getFramebuffer()->width(), atcg::Renderer::getFramebuffer()->height());

        scene->setCamera(camera_controller->getCamera());
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        performance_panel.registerFrameTime(delta_time);
        bool updated = camera_controller->onUpdate(delta_time);

        if(updated)
        {
            integrator->reset();
        }

        atcg::Scripting::handleScriptUpdates(scene, delta_time);

        atcg::Renderer::clear();

        if(atcg::VR::isVRAvailable())
        {
            atcg::ref_ptr<atcg::VRController> controller =
                std::dynamic_pointer_cast<atcg::VRController>(camera_controller);

            if(controller->inMovement())
            {
                atcg::VR::setMovementLine(controller->getControllerPosition(), controller->getControllerIntersection());
            }

            auto [t_left, t_right] = atcg::VR::getRenderTargets();

            t_left->use();
            atcg::Renderer::setViewport(0, 0, atcg::VR::width(), atcg::VR::height());

            atcg::Renderer::clear();

            atcg::Renderer::draw(scene, controller->getCameraLeft());

            atcg::Renderer::drawCameras(scene, controller->getCameraLeft());
            atcg::Renderer::drawLights(scene, controller->getCameraLeft());

            atcg::Renderer::drawCADGrid(controller->getCameraLeft());

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraLeft());
            }

            t_right->use();

            atcg::Renderer::clear();

            atcg::Renderer::draw(scene, controller->getCameraRight());

            atcg::Renderer::drawCameras(scene, controller->getCameraRight());
            atcg::Renderer::drawLights(scene, controller->getCameraRight());

            atcg::Renderer::drawCADGrid(controller->getCameraRight());

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraRight());
            }

            atcg::Renderer::useScreenBuffer();
            atcg::Renderer::setDefaultViewport();

            atcg::VR::renderToScreen();
        }
        else
        {
            atcg::Renderer::clear();

            atcg::Renderer::draw(scene, camera_controller->getCamera());

            atcg::Renderer::drawCameras(scene, camera_controller->getCamera());
            atcg::Renderer::drawLights(scene, camera_controller->getCamera());

            atcg::Renderer::drawCADGrid(camera_controller->getCamera());
        }

        integrator->generateRays(camera_controller->getCamera(), {output_tensor});
        atcg::Framebuffer::useDefault();
        atcg::Renderer::getFramebuffer()->getColorAttachement(0)->setData(output_tensor);
        atcg::Renderer::useScreenBuffer();
        // output_texture->setData(output_tensor);


        uint32_t current_revision = atcg::RevisionStack::numUndos();
        if(current_revision != last_revision)
        {
            last_revision = current_revision;
            initializePathtracer();
        }
    }

#ifndef ATCG_HEADLESS
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
                atcg::Renderer::toggleMSAA(msaa_enabled);
            }

            if(ImGui::BeginCombo("MSAA Samples", combo_preview_value))
            {
                for(int n = 0; n < IM_ARRAYSIZE(msaa_samples); n++)
                {
                    const bool is_selected = (current_msaa_selection_index == n);
                    if(ImGui::Selectable(msaa_samples_str[n], is_selected))
                    {
                        current_msaa_selection_index = n;
                        atcg::Renderer::setMSAA(msaa_samples[current_msaa_selection_index]);
                    }

                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if(is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::End();
        }

        performance_panel.renderPanel(show_performance);
        panel.renderPanel();
        hovered_entity = panel.getSelectedEntity();

        atcg::drawGuizmo(scene, hovered_entity, current_operation, camera_controller->getCamera());
    }
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::Scripting::handleScriptEvents(scene, event);

        atcg::EventDispatcher dispatcher(event);
#ifndef ATCG_HEADLESS
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onKeyPressed));
#endif
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(PBRLayer::onViewportResized));
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
                atcg::Scripting::handleScriptReloads(scene);
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
#endif

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity hovered_entity;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::ref_ptr<atcg::Graph> plane;

    atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler> panel;
    atcg::PerformancePanel performance_panel;
    bool show_performance = false;

    float time       = 0.0f;
    bool in_viewport = false;

    glm::vec2 mouse_pos;

    bool show_render_settings = false;
    bool vsync                = true;

    uint32_t msaa_samples[6]              = {1, 2, 4, 8, 16, 32};
    const char* msaa_samples_str[6]       = {"1", "2", "4", "8", "16", "32"};
    uint32_t current_msaa_selection_index = 3;
    bool msaa_enabled                     = true;
#ifndef ATCG_HEADLESS
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
#endif
    OptixDeviceContext optx_context;
    atcg::ref_ptr<atcg::RayTracingPipeline> pipeline;
    atcg::ref_ptr<atcg::ShaderBindingTable> sbt;
    atcg::ref_ptr<atcg::PathtracingIntegrator> integrator;

    torch::Tensor output_tensor;
    atcg::ref_ptr<atcg::Texture2D> output_texture;

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