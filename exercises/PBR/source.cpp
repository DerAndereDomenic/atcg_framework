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

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FirstPersonController>(aspect_ratio);

        // Load textures
        atcg::ref_ptr<atcg::Texture2D> diffuse_textue;
        {
            auto image = atcg::IO::imread("res/pbr/diffuse.png", 2.2f);
            ATCG_INFO("{0} x {1} x {2}", image->width(), image->height(), image->channels());

            atcg::TextureSpecification spec;
            spec.width     = image->width();
            spec.height    = image->height();
            diffuse_textue = atcg::Texture2D::create(image->data(), spec);
        }
        //
        atcg::ref_ptr<atcg::Texture2D> normal_texture;
        {
            auto image = atcg::IO::imread("res/pbr/normals.png");
            ATCG_INFO("{0} x {1} x {2}", image->width(), image->height(), image->channels());

            atcg::TextureSpecification spec;
            spec.width     = image->width();
            spec.height    = image->height();
            normal_texture = atcg::Texture2D::create(image->data(), spec);
        }

        atcg::ref_ptr<atcg::Texture2D> roughness_texture;
        {
            auto image = atcg::IO::imread("res/pbr/roughness.png");
            ATCG_INFO("{0} x {1} x {2}", image->width(), image->height(), image->channels());

            atcg::TextureSpecification spec;
            spec.width        = image->width();
            spec.height       = image->height();
            roughness_texture = atcg::Texture2D::create(image->data(), spec);
        }

        atcg::ref_ptr<atcg::Texture2D> metallic_texture;
        {
            auto image = atcg::IO::imread("res/pbr/metallic.png");
            ATCG_INFO("{0} x {1} x {2}", image->width(), image->height(), image->channels());

            atcg::TextureSpecification spec;
            spec.width       = image->width();
            spec.height      = image->height();
            spec.format      = atcg::TextureFormat::RINT8;
            metallic_texture = atcg::Texture2D::create(image->data(), spec);
        }

        atcg::ref_ptr<atcg::Texture2D> displacement_texture;
        {
            auto image = atcg::IO::imread("res/pbr/displacement.png");
            ATCG_INFO("{0} x {1} x {2}", image->width(), image->height(), image->channels());

            atcg::TextureSpecification spec;
            spec.width           = image->width();
            spec.height          = image->height();
            spec.format          = atcg::TextureFormat::RINT8;
            displacement_texture = atcg::Texture2D::create(image->data(), spec);
        }

        scene = atcg::make_ref<atcg::Scene>();

        auto options = OpenMesh::IO::Options(OpenMesh::IO::Options::VertexTexCoord);
        plane        = atcg::IO::read_mesh("res/plane.obj", options);

        auto entity = scene->createEntity("Plane");
        entity.addComponent<atcg::TransformComponent>();
        entity.addComponent<atcg::GeometryComponent>(plane);
        auto& renderer  = entity.addComponent<atcg::MeshRenderComponent>();
        renderer.shader = atcg::ShaderManager::getShader("pbr");
        auto& material  = entity.addComponent<atcg::MaterialComponent>();
        material.setDiffuseTexture(diffuse_textue);
        material.setNormalTexture(normal_texture);
        material.setRoughnessTexture(roughness_texture);
        material.setMetallicTexture(metallic_texture);
        material.setDisplacementTexture(displacement_texture);

        panel = atcg::SceneHierarchyPanel(scene);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::drawCADGrid(camera_controller->getCamera());

        atcg::Renderer::draw(scene, camera_controller->getCamera());

        atcg::Renderer::drawCameras(scene, camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("File")) { ImGui::EndMenu(); }

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
        if(event->getKeyCode() == GLFW_KEY_T) { current_operation = ImGuizmo::OPERATION::TRANSLATE; }
        if(event->getKeyCode() == GLFW_KEY_R) { current_operation = ImGuizmo::OPERATION::ROTATE; }
        if(event->getKeyCode() == GLFW_KEY_S) { current_operation = ImGuizmo::OPERATION::SCALE; }
        // if(event->getKeyCode() == GLFW_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

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

    atcg::ref_ptr<atcg::FirstPersonController> camera_controller;

    atcg::ref_ptr<atcg::Graph> plane;

    atcg::SceneHierarchyPanel panel;

    float time       = 0.0f;
    bool in_viewport = false;

    glm::vec2 mouse_pos;

    bool show_render_settings             = false;
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
};

class PBR : public atcg::Application
{
public:
    PBR() : atcg::Application() { pushLayer(new PBRLayer("Layer")); }

    ~PBR() {}
};

atcg::Application* atcg::createApplication()
{
    return new PBR;
}