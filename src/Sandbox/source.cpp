#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <algorithm>


class SandboxLayer : public atcg::Layer
{
public:
    SandboxLayer(const std::string& name) : atcg::Layer(name) {}

    void drawLightSource(atcg::Entity entity, const atcg::ref_ptr<atcg::Camera>& camera)
    {
        atcg::TransformComponent transform = entity.getComponent<atcg::TransformComponent>();

        atcg::Renderer::drawCircle(transform.getPosition(), 0.1f, 0.2f, glm::vec3(1), camera);
    }

    atcg::ref_ptr<atcg::Texture2D> createWhiteNoiseTexture2D(glm::ivec2 dim)
    {
        atcg::ref_ptr<atcg::Shader> compute_shader = atcg::ShaderManager::getShader("white_noise_2D");

        atcg::TextureSpecification spec;
        spec.width                            = dim.x;
        spec.height                           = dim.y;
        spec.format                           = atcg::TextureFormat::RFLOAT;
        atcg::ref_ptr<atcg::Texture2D> result = atcg::Texture2D::create(spec);

        result->useForCompute();

        // Use 8x8x1 = 64 thread sized work group
        compute_shader->dispatch(glm::ivec3(ceil(dim.x / 8), ceil(dim.y / 8), 1));

        return result;
    }

    atcg::ref_ptr<atcg::Texture3D> createWhiteNoiseTexture3D(glm::ivec3 dim)
    {
        atcg::ref_ptr<atcg::Shader> compute_shader = atcg::ShaderManager::getShader("white_noise_3D");

        atcg::TextureSpecification spec;
        spec.width                            = dim.x;
        spec.height                           = dim.y;
        spec.depth                            = dim.z;
        spec.format                           = atcg::TextureFormat::RFLOAT;
        atcg::ref_ptr<atcg::Texture3D> result = atcg::Texture3D::create(spec);

        result->useForCompute();

        // Use 4x4x4 = 64 thread sized work group
        compute_shader->dispatch(glm::ivec3(ceil(dim.x / 4), ceil(dim.y / 4), ceil(dim.z / 4)));

        return result;
    }

    atcg::ref_ptr<atcg::Texture2D> createWorleyNoiseTexture2D(glm::ivec2 dim, uint32_t num_points)
    {
        atcg::ref_ptr<atcg::Shader> compute_shader = atcg::ShaderManager::getShader("worly_noise_2D");

        atcg::TextureSpecification spec;
        spec.width                            = dim.x;
        spec.height                           = dim.y;
        spec.format                           = atcg::TextureFormat::RFLOAT;
        atcg::ref_ptr<atcg::Texture2D> result = atcg::Texture2D::create(spec);


        // Use 8x8x1 = 64 thread sized work group
        compute_shader->setInt("num_points", num_points);
        compute_shader->use();
        result->useForCompute();
        compute_shader->dispatch(glm::ivec3(ceil(dim.x / 8), ceil(dim.y / 8), 1));

        return result;
    }

    atcg::ref_ptr<atcg::Texture3D> createWorleyNoiseTexture3D(glm::ivec3 dim, uint32_t num_points)
    {
        atcg::ref_ptr<atcg::Shader> compute_shader = atcg::ShaderManager::getShader("worly_noise_3D");

        atcg::TextureSpecification spec;
        spec.width                            = dim.x;
        spec.height                           = dim.y;
        spec.depth                            = dim.z;
        spec.format                           = atcg::TextureFormat::RFLOAT;
        atcg::ref_ptr<atcg::Texture3D> result = atcg::Texture3D::create(spec);


        // Use 8x8x1 = 64 thread sized work group
        compute_shader->setInt("num_points", num_points);
        compute_shader->use();
        result->useForCompute();
        compute_shader->dispatch(glm::ivec3(ceil(dim.x / 4), ceil(dim.y / 4), ceil(dim.z / 4)));

        return result;
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Renderer::setPointSize(2.0f);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        atcg::CameraIntrinsics intrinsics;
        intrinsics.setAspectRatio(aspect_ratio);
        camera_controller = atcg::make_ref<atcg::FocusedController>(
            atcg::make_ref<atcg::PerspectiveCamera>(atcg::CameraExtrinsics(), intrinsics));

        cube = atcg::IO::read_mesh((atcg::resource_directory() / "cube.obj").string());

        atcg::ShaderManager::addShader("volume",
                                       atcg::make_ref<atcg::Shader>("src/Sandbox/volume.vs", "src/Sandbox/volume.fs"));
        atcg::ShaderManager::addShader("white_noise_2D",
                                       atcg::make_ref<atcg::Shader>("src/Sandbox/white_noise_2D.glsl"));
        atcg::ShaderManager::addShader("white_noise_3D",
                                       atcg::make_ref<atcg::Shader>("src/Sandbox/white_noise_3D.glsl"));
        atcg::ShaderManager::addShader("worly_noise_2D",
                                       atcg::make_ref<atcg::Shader>("src/Sandbox/worly_noise_2D.glsl"));
        atcg::ShaderManager::addShader("worly_noise_3D",
                                       atcg::make_ref<atcg::Shader>("src/Sandbox/worly_noise_3D.glsl"));


        noise_texture = createWorleyNoiseTexture3D(glm::ivec3(128), num_points);

        scene       = atcg::make_ref<atcg::Scene>();
        cube_entity = scene->createEntity();
        cube_entity.addComponent<atcg::TransformComponent>();
        cube_entity.addComponent<atcg::GeometryComponent>(cube);
        cube_entity.addComponent<atcg::MeshRenderComponent>(atcg::ShaderManager::getShader("volume"));

        light_entity = scene->createEntity();
        light_entity.addComponent<atcg::TransformComponent>();
        light_entity.addComponent<atcg::CustomRenderComponent>(ATCG_BIND_EVENT_FN(SandboxLayer::drawLightSource),
                                                               atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE);

        selected_entity = light_entity;
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::drawCADGrid(camera_controller->getCamera());
        glm::vec3 light_pos = light_entity.getComponent<atcg::TransformComponent>().getPosition();
        atcg::ShaderManager::getShader("volume")->setInt("noise_texture", 0);
        atcg::ShaderManager::getShader("volume")->setVec3("light_position", light_pos);
        atcg::ShaderManager::getShader("volume")->setFloat("sigma_s_base", sigma_s_base);
        atcg::ShaderManager::getShader("volume")->setFloat("sigma_a_base", sigma_a_base);
        atcg::ShaderManager::getShader("volume")->setFloat("g", g);
        noise_texture->use();

        atcg::Dictionary context;
        context.setValue<atcg::ref_ptr<atcg::Camera>>("camera", camera_controller->getCamera());
        scene->draw(context);

        dt = delta_time;
    }

#ifndef ATCG_HEADLESS
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

            if(ImGui::SliderInt("Number of points", reinterpret_cast<int*>(&num_points), 1, 512))
            {
                noise_texture = createWorleyNoiseTexture3D(glm::ivec3(128), num_points);
            }

            ImGui::InputFloat("sigma_s_base", &sigma_s_base);

            ImGui::InputFloat("sigma_a_base", &sigma_a_base);

            ImGui::SliderFloat("g", &g, -0.9999f, 0.9999f);

            ImGui::End();
        }

        ImGui::Begin("Scene Panel");

        if(ImGui::Button("Light"))
        {
            selected_entity = light_entity;
        }

        if(ImGui::Button("Volume"))
        {
            selected_entity = cube_entity;
        }

        ImGui::End();

        // Gizmo test
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();

        const auto& window   = atcg::Application::get()->getWindow();
        glm::vec2 window_pos = window->getPosition();
        ImGuizmo::SetRect(window_pos.x, window_pos.y, (float)window->getWidth(), (float)window->getHeight());

        glm::mat4 camera_projection = camera_controller->getCamera()->getProjection();
        glm::mat4 camera_view       = camera_controller->getCamera()->getView();

        glm::mat4 transform =
            selected_entity.getComponent<atcg::TransformComponent>().getModel();    // sphere->getModel();

        ImGuizmo::Manipulate(glm::value_ptr(camera_view),
                             glm::value_ptr(camera_projection),
                             current_operation,
                             ImGuizmo::LOCAL,
                             glm::value_ptr(transform));

        selected_entity.getComponent<atcg::TransformComponent>().setModel(transform);

        // if(ImGuizmo::IsUsing()) { sphere->setModel(transform); }
    }
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
#ifndef ATCG_HEADLESS
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(SandboxLayer::onKeyPressed));
#endif
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
#endif

private:
    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity cube_entity;
    atcg::Entity light_entity;
    atcg::Entity selected_entity;
    atcg::ref_ptr<atcg::FocusedController> camera_controller;
    atcg::ref_ptr<atcg::Graph> cube;

    atcg::ref_ptr<atcg::Texture3D> noise_texture;

    uint32_t num_points = 16;

    bool show_render_settings = true;

    float sigma_s_base = 20.0f;
    float sigma_a_base = 0.0f;
    float g            = 0.0f;
    float dt           = 1.0f / 60.0f;

#ifndef ATCG_HEADLESS
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
#endif
};

class Sandbox : public atcg::Application
{
public:
    Sandbox() : atcg::Application() { pushLayer(new SandboxLayer("Layer")); }

    ~Sandbox() {}
};

atcg::Application* atcg::createApplication()
{
    return new Sandbox;
}