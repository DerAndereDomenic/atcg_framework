#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <random>


class GrainLayer : public atcg::Layer
{
public:
    GrainLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FocusedController>(aspect_ratio);

        atcg::ref_ptr<atcg::Graph> grain = atcg::IO::read_mesh("../Meshes/rice_low.obj");
        atcg::ref_ptr<atcg::Graph> bowl  = atcg::IO::read_mesh("../Meshes/bowl.obj");

        std::ifstream transform_file("../CVAE/data/Transforms_big.bin", std::ios::in | std::ios::binary);
        std::vector<char> transform_buffer(std::istreambuf_iterator<char>(transform_file), {});
        float* transforms = reinterpret_cast<float*>(transform_buffer.data());

        size_t num_instances =
            transform_buffer.size() / (sizeof(float) * 4 * 4);    // 16 floats for a transformation matrix

        std::vector<atcg::Instance> instances;

        for(int i = 0; i < num_instances; ++i)
        {
            float* current_transform = transforms + 4 * 4 * i;
            glm::mat4 transform      = glm::transpose(glm::make_mat4(current_transform));

            transform[0] *= 10.0f;
            transform[1] *= 10.0f;
            transform[2] *= 10.0f;

            atcg::Instance instance = {transform, glm::vec3(1.2903, 0.558, 0.1328)};

            instances.push_back(instance);
        }

        scene = atcg::make_ref<atcg::Scene>();

        {
            atcg::Entity entity = scene->createEntity();
            auto& transform     = entity.addComponent<atcg::TransformComponent>();
            transform.setPosition(glm::vec3(0, 50, 0));
            entity.addComponent<atcg::GeometryComponent>(grain);
            entity.addComponent<atcg::InstanceRenderComponent>(instances);
        }

        {
            atcg::Entity entity = scene->createEntity();
            auto& transform     = entity.addComponent<atcg::TransformComponent>();
            transform.setPosition(glm::vec3(0, 50, 0));
            entity.addComponent<atcg::GeometryComponent>(bowl);
            entity.addComponent<atcg::MeshRenderComponent>();
        }
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::drawCADGrid(camera_controller->getCamera());

        atcg::Renderer::draw(scene, camera_controller->getCamera());

        dt = delta_time;
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            std::stringstream ss;
            ss << "FPS: " << 1.0f / dt << " | " << dt << " ms\n";
            ImGui::Text(ss.str().c_str());

            ImGui::End();
        }


        // Gizmo test
        // ImGuizmo::SetOrthographic(false);
        // ImGuizmo::BeginFrame();

        // const auto& window   = atcg::Application::get()->getWindow();
        // glm::vec2 window_pos = window->getPosition();
        // ImGuizmo::SetRect(window_pos.x, window_pos.y, (float)window->getWidth(), (float)window->getHeight());

        // glm::mat4 camera_projection = camera_controller->getCamera()->getProjection();
        // glm::mat4 camera_view       = camera_controller->getCamera()->getView();

        // glm::mat4 transform =
        //     selected_entity.getComponent<atcg::TransformComponent>().getModel();    // sphere->getModel();

        // ImGuizmo::Manipulate(glm::value_ptr(camera_view),
        //                      glm::value_ptr(camera_projection),
        //                      current_operation,
        //                      ImGuizmo::LOCAL,
        //                      glm::value_ptr(transform));

        // selected_entity.getComponent<atcg::TransformComponent>().setModel(transform);

        // if(ImGuizmo::IsUsing()) { sphere->setModel(transform); }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(GrainLayer::onKeyPressed));
    }

    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == GLFW_KEY_T) { current_operation = ImGuizmo::OPERATION::TRANSLATE; }
        if(event->getKeyCode() == GLFW_KEY_R) { current_operation = ImGuizmo::OPERATION::ROTATE; }
        if(event->getKeyCode() == GLFW_KEY_S) { current_operation = ImGuizmo::OPERATION::SCALE; }
        // if(event->getKeyCode() == GLFW_KEY_L) { camera_controller->getCamera()->setLookAt(sphere->getPosition()); }

        return true;
    }

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::ref_ptr<atcg::FocusedController> camera_controller;

    bool show_render_settings = true;

    float dt = 1.0f / 60.0f;

    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
};

class Grains : public atcg::Application
{
public:
    Grains() : atcg::Application() { pushLayer(new GrainLayer("Layer")); }

    ~Grains() {}
};

atcg::Application* atcg::createApplication()
{
    return new Grains;
}