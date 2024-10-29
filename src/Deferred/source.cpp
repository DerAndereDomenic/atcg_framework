#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <algorithm>

#include <random>
#include <stb_image.h>

struct RenderContext
{
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::ref_ptr<atcg::Camera> camera;
};

struct GeometryPassData
{
    atcg::ref_ptr<atcg::Shader> geometry_pass_shader;
};

struct LightPassData
{
    atcg::ref_ptr<atcg::Shader> light_pass_shader;
    atcg::ref_ptr<atcg::Graph> screen_quad;
};

class DeferredLayer : public atcg::Layer
{
public:
    DeferredLayer(const std::string& name) : atcg::Layer(name) {}

    void createRenderGraph()
    {
        context         = atcg::make_ref<RenderContext>();
        context->scene  = scene;
        context->camera = camera_controller->getCamera();
        graph           = atcg::make_ref<atcg::RenderGraph<RenderContext>>(context);

        auto [geometry_handle, geometry_builder] = graph->addRenderPass<GeometryPassData, atcg::Framebuffer>();

        geometry_builder->setSetupFunction(
            [](const atcg::ref_ptr<RenderContext>& context,
               const atcg::ref_ptr<GeometryPassData>& data,
               atcg::ref_ptr<atcg::Framebuffer>& framebuffer)
            {
                uint32_t width  = atcg::Renderer::getFramebuffer()->width();
                uint32_t height = atcg::Renderer::getFramebuffer()->height();

                // Setup the G-Buffer
                atcg::TextureSpecification spec_float3;
                spec_float3.width  = width;
                spec_float3.height = height;
                spec_float3.format = atcg::TextureFormat::RGBFLOAT;
                auto position_map  = atcg::Texture2D::create(spec_float3);
                auto normal_map    = atcg::Texture2D::create(spec_float3);
                auto color_map     = atcg::Texture2D::create(spec_float3);

                atcg::TextureSpecification spec_float2;
                spec_float2.width  = width;
                spec_float2.height = height;
                spec_float2.format = atcg::TextureFormat::RGFLOAT;
                auto spec_met_map  = atcg::Texture2D::create(spec_float2);

                framebuffer = atcg::make_ref<atcg::Framebuffer>(width, height);
                framebuffer->attachTexture(position_map);
                framebuffer->attachTexture(normal_map);
                framebuffer->attachTexture(color_map);
                framebuffer->attachTexture(spec_met_map);
                framebuffer->attachDepth();
                framebuffer->complete();

                // Load the geometry pass shader
                std::string vertex_path    = std::string(ATCG_TARGET_DIR) + "/geometry.vs";
                std::string fragment_path  = std::string(ATCG_TARGET_DIR) + "/geometry.fs";
                data->geometry_pass_shader = atcg::make_ref<atcg::Shader>(vertex_path, fragment_path);
            });

        geometry_builder->setRenderFunction(
            [](const atcg::ref_ptr<RenderContext>& context,
               const std::vector<std::any>& inputs,
               const atcg::ref_ptr<GeometryPassData>& data,
               const atcg::ref_ptr<atcg::Framebuffer>& framebuffer)
            {
                // auto framebuffer = std::any_cast<atcg::ref_ptr<atcg::Framebuffer>>(inputs[0]);
                framebuffer->use();
                atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
                atcg::Renderer::clear();

                const auto& view = context->scene->getAllEntitiesWith<atcg::TransformComponent,
                                                                      atcg::GeometryComponent,
                                                                      atcg::MeshRenderComponent>();

                for(auto e: view)
                {
                    atcg::Entity entity(e, context->scene.get());

                    auto& transform = entity.getComponent<atcg::TransformComponent>();
                    auto& geometry  = entity.getComponent<atcg::GeometryComponent>();
                    auto& renderer  = entity.getComponent<atcg::MeshRenderComponent>();

                    atcg::Renderer::draw(geometry.graph,
                                         renderer.material,
                                         context->camera,
                                         transform.getModel(),
                                         glm::vec3(1),
                                         data->geometry_pass_shader);
                }
            });

        auto [light_handle, light_builder] = graph->addRenderPass<LightPassData, atcg::Framebuffer>();

        light_builder->setSetupFunction(
            [](const atcg::ref_ptr<RenderContext>& context,
               const atcg::ref_ptr<LightPassData>& data,
               atcg::ref_ptr<atcg::Framebuffer>& framebuffer)
            {
                framebuffer = atcg::Renderer::getFramebuffer();

                // Load the geometry pass shader
                std::string vertex_path   = std::string(ATCG_TARGET_DIR) + "/light.vs";
                std::string fragment_path = std::string(ATCG_TARGET_DIR) + "/light.fs";
                data->light_pass_shader   = atcg::make_ref<atcg::Shader>(vertex_path, fragment_path);

                std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                                      atcg::Vertex(glm::vec3(1, -1, 0)),
                                                      atcg::Vertex(glm::vec3(1, 1, 0)),
                                                      atcg::Vertex(glm::vec3(-1, 1, 0))};

                std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

                data->screen_quad = atcg::Graph::createTriangleMesh(vertices, edges);
            });

        light_builder->setRenderFunction(
            [](const atcg::ref_ptr<RenderContext>& context,
               const std::vector<std::any>& inputs,
               const atcg::ref_ptr<LightPassData>& data,
               const atcg::ref_ptr<atcg::Framebuffer>& framebuffer)
            {
                auto g_buffer = std::any_cast<atcg::ref_ptr<atcg::Framebuffer>>(inputs[0]);

                framebuffer->use();
                atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
                atcg::Renderer::clear();

                atcg::Renderer::drawSkybox(context->camera);

                framebuffer->blit(g_buffer, false);

                data->light_pass_shader->setInt("position_texture", 9);
                data->light_pass_shader->setInt("normal_texture", 10);
                data->light_pass_shader->setInt("color_texture", 11);
                data->light_pass_shader->setInt("spec_met_texture", 12);
                data->light_pass_shader->setVec3("camera_pos", context->camera->getPosition());
                data->light_pass_shader->setVec3("camera_dir", context->camera->getDirection());
                g_buffer->getColorAttachement(0)->use(9);
                g_buffer->getColorAttachement(1)->use(10);
                g_buffer->getColorAttachement(2)->use(11);
                g_buffer->getColorAttachement(3)->use(12);
                atcg::Renderer::toggleDepthTesting(false);
                atcg::Renderer::draw(data->screen_quad, {}, glm::mat4(1), glm::vec3(1), data->light_pass_shader);
                atcg::Renderer::toggleDepthTesting(true);

                // const auto& view = context->scene->getAllEntitiesWith<atcg::TransformComponent>();

                // for(auto e: view)
                // {
                //     atcg::Entity entity(e, context->scene.get());

                //     if(entity.hasAnyComponent<atcg::PointRenderComponent,
                //                               atcg::EdgeRenderComponent,
                //                               atcg::EdgeCylinderRenderComponent>())
                //     {
                //         atcg::Renderer::draw(entity, context->camera);
                //     }
                // }
            });

        graph->addDependency(geometry_handle, light_handle);

        graph->compile();
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

        auto skybox = atcg::IO::imread((atcg::resource_directory() / "pbr/skybox.hdr").string());
        ATCG_TRACE("{0} {1} {2}", skybox->width(), skybox->height(), skybox->channels());
        atcg::Renderer::setSkybox(skybox);

        scene = atcg::IO::read_scene((atcg::resource_directory() / "test_scene.obj").string());

        panel = atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>(scene);

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FirstPersonController>(aspect_ratio);

        createRenderGraph();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        graph->execute();

        // atcg::Renderer::drawCameras(scene, camera_controller->getCamera());

        // atcg::Renderer::drawCADGrid(camera_controller->getCamera());
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

        panel.renderPanel();
        hovered_entity = panel.getSelectedEntity();

        atcg::drawGuizmo(hovered_entity, current_operation, camera_controller->getCamera());
    }
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
#ifndef ATCG_HEADLESS
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(DeferredLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(DeferredLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(DeferredLayer::onKeyPressed));
#endif
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(DeferredLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);

        createRenderGraph();
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
            current_operation = ImGuizmo::OPERATION::ROTATE;
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
    atcg::ref_ptr<RenderContext> context;
    atcg::ref_ptr<atcg::RenderGraph<RenderContext>> graph;

    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity hovered_entity;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::ref_ptr<atcg::Graph> plane;

    atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler> panel;

    float time       = 0.0f;
    bool in_viewport = false;

    glm::vec2 mouse_pos;

    bool show_render_settings = false;
#ifndef ATCG_HEADLESS
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
#endif
};

class Deferred : public atcg::Application
{
public:
    Deferred(const atcg::WindowProps& props) : atcg::Application(props) { pushLayer(new DeferredLayer("Layer")); }

    ~Deferred() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = true;
    return new Deferred(props);
}