#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <random>

class MandelbulbLayer : public atcg::Layer
{
public:
    MandelbulbLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(false);
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FirstPersonController>(aspect_ratio);

        auto shader = atcg::make_ref<atcg::Shader>("src/Mandelbulb/Mandelbulb.glsl");
        atcg::ShaderManager::addShader("Mandelbulb", shader);

        atcg::TextureSpecification spec;
        spec.width  = window->getWidth();
        spec.height = window->getHeight();
        spec.format = atcg::TextureFormat::RGBAFLOAT;
        texture     = atcg::Texture2D::create(spec);

        std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f)),
                                              atcg::Vertex(glm::vec3(1.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                                              atcg::Vertex(glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f)),
                                              atcg::Vertex(glm::vec3(-1.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f))};

        std::vector<glm::u32vec3> indices = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

        quad_mesh = atcg::Graph::createTriangleMesh(vertices, indices);

        screen_id = atcg::Renderer::popTextureID();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::ShaderManager::getShader("Mandelbulb")
            ->setVec3("camera_position", camera_controller->getCamera()->getPosition());
        atcg::ShaderManager::getShader("Mandelbulb")->setMat4("V", camera_controller->getCamera()->getView());
        atcg::ShaderManager::getShader("Mandelbulb")->setMat4("P", camera_controller->getCamera()->getProjection());
        atcg::ShaderManager::getShader("Mandelbulb")
            ->setMat4("VP", camera_controller->getCamera()->getViewProjection());
        atcg::ShaderManager::getShader("Mandelbulb")
            ->setMat4("invVP", glm::inverse(camera_controller->getCamera()->getViewProjection()));
        atcg::ShaderManager::getShader("Mandelbulb")->use();
        texture->useForCompute();
        atcg::ShaderManager::getShader("Mandelbulb")
            ->dispatch(glm::ivec3(ceil(texture->width() / 8), ceil(texture->height() / 8), 1));

        atcg::ShaderManager::getShader("screen")->setInt("screen_texture", screen_id);
        atcg::ShaderManager::getShader("screen")->use();

        texture->use(screen_id);

        atcg::Renderer::draw(quad_mesh, {}, glm::mat4(1), glm::vec3(1), atcg::ShaderManager::getShader("screen"));
    }

    virtual void onImGuiRender() override {}

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(MandelbulbLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);
        atcg::TextureSpecification spec;
        spec.width  = event->getWidth();
        spec.height = event->getHeight();
        spec.format = atcg::TextureFormat::RGBAFLOAT;
        texture     = atcg::Texture2D::create(spec);
        return false;
    }


private:
    atcg::ref_ptr<atcg::FirstPersonController> camera_controller;

    atcg::ref_ptr<atcg::Texture2D> texture;
    atcg::ref_ptr<atcg::Graph> quad_mesh;

    int screen_id;
};

class Mandelbulb : public atcg::Application
{
public:
    Mandelbulb() : atcg::Application() { pushLayer(new MandelbulbLayer("Layer")); }

    ~Mandelbulb() {}
};

atcg::Application* atcg::createApplication()
{
    return new Mandelbulb;
}