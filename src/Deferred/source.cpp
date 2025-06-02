#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <algorithm>

#include <random>
#include <stb_image.h>

class DeferredLayer : public atcg::Layer
{
public:
    DeferredLayer(const std::string& name) : atcg::Layer(name) {}

    void createRenderGraph()
    {
        atcg::Dictionary context;
        graph = atcg::make_ref<atcg::RenderGraph>();

        auto [geometry_handle, geometry_builder] = graph->addRenderPass("Geometry Pass");

        geometry_builder
            ->setSetupFunction(
                [](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
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

                    atcg::TextureSpecification spec_int;
                    spec_int.width               = width;
                    spec_int.height              = height;
                    spec_int.format              = atcg::TextureFormat::RINT;
                    spec_int.sampler.filter_mode = atcg::TextureFilterMode::NEAREST;
                    auto entity_map              = atcg::Texture2D::create(spec_int);

                    auto framebuffer = atcg::make_ref<atcg::Framebuffer>(width, height);
                    framebuffer->attachTexture(position_map);
                    framebuffer->attachTexture(normal_map);
                    framebuffer->attachTexture(color_map);
                    framebuffer->attachTexture(spec_met_map);
                    framebuffer->attachTexture(entity_map);
                    framebuffer->attachDepth();
                    framebuffer->complete();

                    output.setValue("g_buffer", framebuffer);

                    // Load the geometry pass shader
                    std::string vertex_path   = std::string(ATCG_TARGET_DIR) + "/geometry.vs";
                    std::string fragment_path = std::string(ATCG_TARGET_DIR) + "/geometry.fs";
                    auto geometry_pass_shader = atcg::make_ref<atcg::Shader>(vertex_path, fragment_path);
                    data.setValue("geometry_pass_shader", geometry_pass_shader);
                })
            ->setRenderFunction(
                [](atcg::Dictionary& context,
                   const atcg::Dictionary& inputs,
                   atcg::Dictionary& data,
                   atcg::Dictionary& output)
                {
                    auto framebuffer = output.getValue<atcg::ref_ptr<atcg::Framebuffer>>("g_buffer");
                    framebuffer->use();
                    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
                    atcg::Renderer::setViewport(0, 0, framebuffer->width(), framebuffer->height());
                    atcg::Renderer::clear();
                    int value = -1;
                    framebuffer->getColorAttachement(4)->fill(&value);

                    auto scene       = context.getValue<atcg::Scene*>("scene");
                    const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent,
                                                                 atcg::GeometryComponent,
                                                                 atcg::MeshRenderComponent>();

                    atcg::ComponentRenderer component_renderer;
                    auto camera = context.getValue<atcg::ref_ptr<atcg::Camera>>("camera");

                    auto geometry_pass_shader = data.getValue<atcg::ref_ptr<atcg::Shader>>("geometry_pass_shader");
                    atcg::Dictionary auxiliary;
                    auxiliary.setValue("override_shader", geometry_pass_shader);
                    for(auto e: view)
                    {
                        atcg::Entity entity(e, scene);

                        component_renderer.renderComponent<atcg::MeshRenderComponent>(entity, camera, auxiliary);
                    }
                });

        auto [light_handle, light_builder] = graph->addRenderPass("Lighting Pass");

        light_builder
            ->setSetupFunction(
                [](atcg::Dictionary& context, atcg::Dictionary& data, atcg::Dictionary& output)
                {
                    auto framebuffer = atcg::Renderer::getFramebuffer();
                    output.setValue("framebuffer", framebuffer);

                    // Load the geometry pass shader
                    std::string vertex_path   = std::string(ATCG_TARGET_DIR) + "/light.vs";
                    std::string fragment_path = std::string(ATCG_TARGET_DIR) + "/light.fs";
                    auto light_pass_shader    = atcg::make_ref<atcg::Shader>(vertex_path, fragment_path);
                    data.setValue("light_pass_shader", light_pass_shader);

                    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

                    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

                    auto screen_quad = atcg::Graph::createTriangleMesh(vertices, edges);
                    data.setValue("screen_quad", screen_quad);
                })
            ->setRenderFunction(
                [](atcg::Dictionary& context,
                   const atcg::Dictionary& inputs,
                   atcg::Dictionary& data,
                   atcg::Dictionary& output)
                {
                    auto g_buffer = inputs.getValue<atcg::ref_ptr<atcg::Framebuffer>>("g_buffer");

                    auto framebuffer = output.getValue<atcg::ref_ptr<atcg::Framebuffer>>("framebuffer");

                    framebuffer->use();
                    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
                    atcg::Renderer::setDefaultViewport();
                    atcg::Renderer::clear();

                    auto camera = context.getValue<atcg::ref_ptr<atcg::Camera>>("camera");

                    // TODO skybox
                    auto scene = context.getValue<atcg::Scene*>("scene");
                    atcg::Renderer::drawSkybox(scene->getSkyboxCubemap(), camera);

                    framebuffer->blit(g_buffer, false);

                    auto light_view = scene->getAllEntitiesWith<atcg::PointLightComponent>();
                    int num_lights  = 0;

                    auto light_pass_shader = data.getValue<atcg::ref_ptr<atcg::Shader>>("light_pass_shader");
                    for(auto e: light_view)
                    {
                        atcg::Entity light(e, scene);
                        auto& light_transform = light.getComponent<atcg::TransformComponent>();
                        auto& point_light     = light.getComponent<atcg::PointLightComponent>();
                        std::stringstream ss;
                        ss << "[" << std::to_string(num_lights) << "]";
                        light_pass_shader->setVec3("light_positions" + ss.str(), light_transform.getPosition());
                        light_pass_shader->setFloat("light_intensities" + ss.str(), point_light.intensity);
                        light_pass_shader->setVec3("light_colors" + ss.str(), point_light.color);
                        ++num_lights;
                    }
                    light_pass_shader->setInt("num_lights", num_lights);


                    uint32_t point_light_id = atcg::Renderer::popTextureID();
                    auto point_light_depth_maps =
                        *inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<atcg::TextureCubeArray>>>("point_light_depth_"
                                                                                               "maps");
                    point_light_depth_maps->use(point_light_id);
                    light_pass_shader->setInt("point_light_depth_maps", point_light_id);

                    uint32_t position_texture_id = atcg::Renderer::popTextureID();
                    light_pass_shader->setInt("position_texture", position_texture_id);
                    uint32_t normal_texture_id = atcg::Renderer::popTextureID();
                    light_pass_shader->setInt("normal_texture", normal_texture_id);
                    uint32_t color_texture_id = atcg::Renderer::popTextureID();
                    light_pass_shader->setInt("color_texture", color_texture_id);
                    uint32_t spec_met_texture_id = atcg::Renderer::popTextureID();
                    light_pass_shader->setInt("spec_met_texture", spec_met_texture_id);
                    uint32_t entity_texture_id = atcg::Renderer::popTextureID();
                    light_pass_shader->setInt("entity_texture", entity_texture_id);
                    light_pass_shader->setVec3("camera_pos", camera->getPosition());
                    light_pass_shader->setVec3("camera_dir", camera->getDirection());

                    g_buffer->getColorAttachement(0)->use(position_texture_id);
                    g_buffer->getColorAttachement(1)->use(normal_texture_id);
                    g_buffer->getColorAttachement(2)->use(color_texture_id);
                    g_buffer->getColorAttachement(3)->use(spec_met_texture_id);
                    g_buffer->getColorAttachement(4)->use(entity_texture_id);

                    auto screen_quad = data.getValue<atcg::ref_ptr<atcg::Graph>>("screen_quad");
                    atcg::Renderer::toggleDepthTesting(false);
                    atcg::Renderer::draw(screen_quad, {}, glm::mat4(1), glm::vec3(1), light_pass_shader);
                    atcg::Renderer::toggleDepthTesting(true);

                    atcg::Renderer::pushTextureID(position_texture_id);
                    atcg::Renderer::pushTextureID(normal_texture_id);
                    atcg::Renderer::pushTextureID(color_texture_id);
                    atcg::Renderer::pushTextureID(spec_met_texture_id);
                    atcg::Renderer::pushTextureID(entity_texture_id);
                    atcg::Renderer::pushTextureID(point_light_id);

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

        auto shadow_handle = graph->addRenderPass(atcg::make_ref<atcg::ShadowPass>());

        // shadow_builder->setSetupFunction(
        //     [](const atcg::ref_ptr<RenderContext>& context,
        //        const atcg::ref_ptr<ShadowPassData>& data,
        //        atcg::ref_ptr<int>&)
        //     {
        //         std::string vertex_path   = std::string(ATCG_TARGET_DIR) + "/depth.vs";
        //         std::string geometry_path = std::string(ATCG_TARGET_DIR) + "/depth.gs";
        //         std::string fragment_path = std::string(ATCG_TARGET_DIR) + "/depth.fs";
        //         data->depth_pass_shader   = atcg::make_ref<atcg::Shader>(vertex_path, fragment_path, geometry_path);
        //     });

        // shadow_builder->setRenderFunction(
        //     [](const atcg::ref_ptr<RenderContext>& context,
        //        const std::vector<std::any>& inputs,
        //        const atcg::ref_ptr<ShadowPassData>& data,
        //        const atcg::ref_ptr<int>&)
        //     {
        //         float n              = 1.0f;
        //         float f              = 25.0f;
        //         glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, n, f);

        //         data->depth_pass_shader->setFloat("far_plane", f);

        //         auto light_view = context->scene->getAllEntitiesWith<PointLightComponent>();
        //         for(auto e: light_view)
        //         {
        //             atcg::Entity entity(e, context->scene.get());

        //             auto& point_light = entity.getComponent<PointLightComponent>();
        //             auto& transform   = entity.getComponent<atcg::TransformComponent>();
        //             point_light.framebuffer->use();
        //             atcg::Renderer::setViewport(0,
        //                                         0,
        //                                         point_light.framebuffer->width(),
        //                                         point_light.framebuffer->height());
        //             atcg::Renderer::clear();

        //             glm::vec3 lightPos = transform.getPosition();
        //             data->depth_pass_shader->setVec3("lightPos", lightPos);
        //             data->depth_pass_shader->setMat4("shadowMatrices[0]",
        //                                              projection * glm::lookAt(lightPos,
        //                                                                       lightPos + glm::vec3(1.0, 0.0, 0.0),
        //                                                                       glm::vec3(0.0, -1.0, 0.0)));
        //             data->depth_pass_shader->setMat4("shadowMatrices[1]",
        //                                              projection * glm::lookAt(lightPos,
        //                                                                       lightPos + glm::vec3(-1.0, 0.0, 0.0),
        //                                                                       glm::vec3(0.0, -1.0, 0.0)));
        //             data->depth_pass_shader->setMat4("shadowMatrices[2]",
        //                                              projection * glm::lookAt(lightPos,
        //                                                                       lightPos + glm::vec3(0.0, 1.0, 0.0),
        //                                                                       glm::vec3(0.0, 0.0, 1.0)));
        //             data->depth_pass_shader->setMat4("shadowMatrices[3]",
        //                                              projection * glm::lookAt(lightPos,
        //                                                                       lightPos + glm::vec3(0.0, -1.0, 0.0),
        //                                                                       glm::vec3(0.0, 0.0, -1.0)));
        //             data->depth_pass_shader->setMat4("shadowMatrices[4]",
        //                                              projection * glm::lookAt(lightPos,
        //                                                                       lightPos + glm::vec3(0.0, 0.0, 1.0),
        //                                                                       glm::vec3(0.0, -1.0, 0.0)));
        //             data->depth_pass_shader->setMat4("shadowMatrices[5]",
        //                                              projection * glm::lookAt(lightPos,
        //                                                                       lightPos + glm::vec3(0.0, 0.0, -1.0),
        //                                                                       glm::vec3(0.0, -1.0, 0.0)));

        //             const auto& view = context->scene->getAllEntitiesWith<atcg::TransformComponent,
        //                                                                   atcg::GeometryComponent,
        //                                                                   atcg::MeshRenderComponent>();

        //             // Draw scene
        //             for(auto e: view)
        //             {
        //                 atcg::Entity entity(e, context->scene.get());

        //                 auto& transform = entity.getComponent<atcg::TransformComponent>();
        //                 auto& geometry  = entity.getComponent<atcg::GeometryComponent>();

        //                 atcg::Renderer::draw(geometry.graph,
        //                                      context->camera,
        //                                      transform.getModel(),
        //                                      glm::vec3(1),
        //                                      data->depth_pass_shader);
        //             }
        //         }
        //     });

        graph->addDependency(geometry_handle, "g_buffer", light_handle, "g_buffer");
        graph->addDependency(shadow_handle, "point_light_depth_maps", light_handle, "point_light_depth_maps");

        graph->compile(context);

        scene->setRenderGraph(graph);
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::toggleMSAA(false);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

        // auto skybox = atcg::IO::imread((atcg::resource_directory() / "pbr/skybox.hdr").string());
        // ATCG_TRACE("{0} {1} {2}", skybox->width(), skybox->height(), skybox->channels());
        // atcg::Renderer::setSkybox(skybox);

        scene = atcg::IO::read_scene((atcg::resource_directory() / "test_scene.obj").string());

        auto point_light = scene->createEntity("Light");
        auto& light      = point_light.addComponent<atcg::PointLightComponent>();
        light.color      = glm::vec3(1);
        light.intensity  = 10.0f;
        point_light.addComponent<atcg::TransformComponent>(glm::vec3(0, 5, 0));

        panel = atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler>(scene);

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        atcg::CameraIntrinsics intrinsics;
        intrinsics.setAspectRatio(aspect_ratio);
        camera_controller = atcg::make_ref<atcg::FirstPersonController>(
            atcg::make_ref<atcg::PerspectiveCamera>(atcg::CameraExtrinsics(), intrinsics));

        createRenderGraph();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Dictionary context;
        context.setValue<atcg::ref_ptr<atcg::Camera>>("camera", camera_controller->getCamera());
        scene->draw(context);

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

        atcg::drawGuizmo(scene, hovered_entity, current_operation, camera_controller->getCamera());
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
    atcg::ref_ptr<atcg::RenderGraph> graph;

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