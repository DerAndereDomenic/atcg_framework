#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>

class FluidSimLayer : public atcg::Layer
{
public:
    FluidSimLayer(const std::string& name) : atcg::Layer(name) {}

    void createTextures(int width, int height)
    {
        output_texture = atcg::TextureBuilder()
                             .setWidth(width)
                             .setHeight(height)
                             .setFormat(atcg::TextureFormat::RGBAFLOAT)
                             .create<atcg::Texture2D>();

        velocity_input = atcg::TextureBuilder()
                             .setWidth(width)
                             .setHeight(height)
                             .setFormat(atcg::TextureFormat::RGFLOAT)
                             .setWrapMode(atcg::TextureWrapMode::BORDER)
                             .create<atcg::Texture2D>();

        pressure_input = atcg::TextureBuilder()
                             .setWidth(width)
                             .setHeight(height)
                             .setFormat(atcg::TextureFormat::RFLOAT)
                             .setWrapMode(atcg::TextureWrapMode::CLAMP_TO_EDGE)
                             .create<atcg::Texture2D>();

        velocity_output = atcg::TextureBuilder()
                              .setWidth(width)
                              .setHeight(height)
                              .setFormat(atcg::TextureFormat::RGFLOAT)
                              .setWrapMode(atcg::TextureWrapMode::BORDER)
                              .create<atcg::Texture2D>();


        pressure_output = atcg::TextureBuilder()
                              .setWidth(width)
                              .setHeight(height)
                              .setFormat(atcg::TextureFormat::RFLOAT)
                              .setWrapMode(atcg::TextureWrapMode::CLAMP_TO_EDGE)
                              .create<atcg::Texture2D>();

        advection_texture = atcg::TextureBuilder()
                                .setWidth(width)
                                .setHeight(height)
                                .setFormat(atcg::TextureFormat::RGFLOAT)
                                .setWrapMode(atcg::TextureWrapMode::BORDER)
                                .create<atcg::Texture2D>();

        divergence_texture = atcg::TextureBuilder()
                                 .setWidth(width)
                                 .setHeight(height)
                                 .setFormat(atcg::TextureFormat::RFLOAT)
                                 .setWrapMode(atcg::TextureWrapMode::BORDER)
                                 .create<atcg::Texture2D>();

        dye_input = atcg::TextureBuilder()
                        .setWidth(width)
                        .setHeight(height)
                        .setFormat(atcg::TextureFormat::RGBAFLOAT)
                        .setWrapMode(atcg::TextureWrapMode::CLAMP_TO_EDGE)
                        .create<atcg::Texture2D>();

        dye_interm = atcg::TextureBuilder()
                         .setWidth(width)
                         .setHeight(height)
                         .setFormat(atcg::TextureFormat::RGBAFLOAT)
                         .setWrapMode(atcg::TextureWrapMode::CLAMP_TO_EDGE)
                         .create<atcg::Texture2D>();

        dye_output = atcg::TextureBuilder()
                         .setWidth(width)
                         .setHeight(height)
                         .setFormat(atcg::TextureFormat::RGBAFLOAT)
                         .setWrapMode(atcg::TextureWrapMode::CLAMP_TO_EDGE)
                         .create<atcg::Texture2D>();
    }

    void initializePass(int width, int height)
    {
        createTextures(width, height);

        velocity_input->useForCompute(0);
        pressure_input->useForCompute(1);
        dye_input->useForCompute(2);
        init_shader->dispatch(glm::ivec3(ceil(width / 8), ceil(height / 8), 1));
    }

    void advectionPass(float delta_time)
    {
        atcg::GraphicsCommand::beginRenderPass(nullptr);

        advection_texture->useForCompute(0);
        advection_shader->setInt("prev_velocity_texture", 0);
        advection_shader->setFloat("delta_time", delta_time);
        advection_shader->setFloat("rho", density);
        atcg::GraphicsCommand::bindTexture(0, velocity_input);
        advection_shader->dispatch(
            glm::ivec3(ceil(output_texture->width() / 8), ceil(output_texture->height() / 8), 1));

        atcg::GraphicsCommand::endRenderPass();
    }

    void divergencePass()
    {
        atcg::GraphicsCommand::beginRenderPass(nullptr);

        divergence_texture->useForCompute(0);
        divergence_shader->setInt("velocity_texture", 0);
        atcg::GraphicsCommand::bindTexture(0, advection_texture);
        divergence_shader->dispatch(
            glm::ivec3(ceil(output_texture->width() / 8), ceil(output_texture->height() / 8), 1));

        atcg::GraphicsCommand::endRenderPass();
    }

    void jacobiPass(float delta_time)
    {
        atcg::GraphicsCommand::beginRenderPass(nullptr);

        const int iterations = 50;

        for(int i = 0; i < iterations; ++i)
        {
            pressure_output->useForCompute(0);
            jacobi_shader->setInt("prev_pressure_texture", 0);
            jacobi_shader->setInt("divergence_texture", 1);
            jacobi_shader->setFloat("delta_time", delta_time);
            jacobi_shader->setFloat("rho", density);
            atcg::GraphicsCommand::bindTexture(0, pressure_input);
            atcg::GraphicsCommand::bindTexture(1, divergence_texture);
            jacobi_shader->dispatch(
                glm::ivec3(ceil(output_texture->width() / 8), ceil(output_texture->height() / 8), 1));

            std::swap(pressure_input, pressure_output);
        }

        std::swap(pressure_input, pressure_output);    // Swap back to ensure pressure_output has the final result

        atcg::GraphicsCommand::endRenderPass();
    }

    void projectionPass(float delta_time)
    {
        atcg::GraphicsCommand::beginRenderPass(nullptr);

        velocity_output->useForCompute(0);
        projection_shader->setInt("advection_texture", 0);
        projection_shader->setInt("pressure_texture", 1);
        projection_shader->setFloat("rho", density);
        projection_shader->setFloat("delta_time", delta_time);
        atcg::GraphicsCommand::bindTexture(0, advection_texture);
        atcg::GraphicsCommand::bindTexture(1, pressure_output);
        projection_shader->dispatch(
            glm::ivec3(ceil(output_texture->width() / 8), ceil(output_texture->height() / 8), 1));

        atcg::GraphicsCommand::endRenderPass();
    }

    void dyeAdvectionPass(float delta_time)
    {
        atcg::GraphicsCommand::beginRenderPass(nullptr);

        dye_interm->useForCompute(0);
        dye_advection_shader->setInt("velocity_texture", 0);
        dye_advection_shader->setInt("prev_dye_texture", 1);
        dye_advection_shader->setFloat("delta_time", delta_time);
        dye_advection_shader->setFloat("rho", density);
        atcg::GraphicsCommand::bindTexture(0, velocity_output);
        atcg::GraphicsCommand::bindTexture(1, dye_input);
        dye_advection_shader->dispatch(
            glm::ivec3(ceil(output_texture->width() / 8), ceil(output_texture->height() / 8), 1));

        atcg::GraphicsCommand::endRenderPass();
    }

    void dyeDiffusionPass(float delta_time)
    {
        atcg::GraphicsCommand::beginRenderPass(nullptr);

        dye_output->useForCompute(0);
        dye_diffusion_shader->setInt("prev_dye_texture", 0);
        dye_diffusion_shader->setFloat("delta_time", delta_time);
        dye_diffusion_shader->setFloat("rho", density);
        atcg::GraphicsCommand::bindTexture(0, dye_interm);
        dye_diffusion_shader->dispatch(
            glm::ivec3(ceil(output_texture->width() / 8), ceil(output_texture->height() / 8), 1));

        atcg::GraphicsCommand::endRenderPass();
    }

    void visualizePass()
    {
        atcg::GraphicsCommand::beginRenderPass(nullptr);

        output_texture->useForCompute(0);
        visualize_shader->setInt("veloctiy", 0);
        visualize_shader->setInt("pressure", 1);
        visualize_shader->setInt("dye", 2);
        atcg::GraphicsCommand::bindTexture(0, velocity_output);
        atcg::GraphicsCommand::bindTexture(1, pressure_output);
        atcg::GraphicsCommand::bindTexture(2, dye_output);
        visualize_shader->dispatch(
            glm::ivec3(ceil(output_texture->width() / 8), ceil(output_texture->height() / 8), 1));

        atcg::GraphicsCommand::endRenderPass();
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(false);
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        atcg::CameraIntrinsics intrinsics;
        intrinsics.setAspectRatio(aspect_ratio);
        camera_controller = atcg::make_ref<atcg::FirstPersonController>(
            atcg::make_ref<atcg::PerspectiveCamera>(atcg::CameraExtrinsics(), intrinsics));

        init_shader          = atcg::make_ref<atcg::Shader>("src/FluidSimulation/Initialize.glsl");
        advection_shader     = atcg::make_ref<atcg::Shader>("src/FluidSimulation/Advection.glsl");
        visualize_shader     = atcg::make_ref<atcg::Shader>("src/FluidSimulation/Visualize.glsl");
        divergence_shader    = atcg::make_ref<atcg::Shader>("src/FluidSimulation/Divergence.glsl");
        jacobi_shader        = atcg::make_ref<atcg::Shader>("src/FluidSimulation/Jacobi.glsl");
        projection_shader    = atcg::make_ref<atcg::Shader>("src/FluidSimulation/Projection.glsl");
        dye_advection_shader = atcg::make_ref<atcg::Shader>("src/FluidSimulation/DyeAdvection.glsl");
        dye_diffusion_shader = atcg::make_ref<atcg::Shader>("src/FluidSimulation/DyeDiffusion.glsl");

        initializePass(window->getWidth(), window->getHeight());

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

        advectionPass(delta_time);

        divergencePass();

        jacobiPass(delta_time);

        projectionPass(delta_time);

        dyeAdvectionPass(delta_time);

        dyeDiffusionPass(delta_time);

        visualizePass();

        atcg::ShaderManager::getShader("screen")->setInt("screen_texture", screen_id);

        atcg::GraphicsPipeline pipeline = atcg::GraphicsPipeline().setShader(atcg::ShaderManager::getShader("screen"));

        atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());
        atcg::GraphicsCommand::clear();
        atcg::GraphicsCommand::bindTexture(screen_id, output_texture);

        atcg::Renderer::drawVAO(quad_mesh->getVerticesArray(), {}, glm::mat4(1), pipeline, quad_mesh->n_vertices());
        atcg::GraphicsCommand::endRenderPass();

        std::swap(velocity_input, velocity_output);
        std::swap(dye_input, dye_output);
    }

    virtual void onImGuiRender() override {}

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(FluidSimLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);
        initializePass(event->getWidth(), event->getHeight());
        return false;
    }


private:
    atcg::ref_ptr<atcg::FirstPersonController> camera_controller;

    atcg::ref_ptr<atcg::Texture2D> output_texture;
    atcg::ref_ptr<atcg::Texture2D> divergence_texture;
    atcg::ref_ptr<atcg::Texture2D> advection_texture;
    atcg::ref_ptr<atcg::Texture2D> velocity_input;
    atcg::ref_ptr<atcg::Texture2D> velocity_output;
    atcg::ref_ptr<atcg::Texture2D> pressure_input;
    atcg::ref_ptr<atcg::Texture2D> pressure_output;
    atcg::ref_ptr<atcg::Texture2D> dye_input;
    atcg::ref_ptr<atcg::Texture2D> dye_interm;
    atcg::ref_ptr<atcg::Texture2D> dye_output;
    atcg::ref_ptr<atcg::Graph> quad_mesh;

    atcg::ref_ptr<atcg::Shader> init_shader;
    atcg::ref_ptr<atcg::Shader> visualize_shader;
    atcg::ref_ptr<atcg::Shader> advection_shader;
    atcg::ref_ptr<atcg::Shader> divergence_shader;
    atcg::ref_ptr<atcg::Shader> jacobi_shader;
    atcg::ref_ptr<atcg::Shader> projection_shader;
    atcg::ref_ptr<atcg::Shader> dye_advection_shader;
    atcg::ref_ptr<atcg::Shader> dye_diffusion_shader;

    float density = 1.0f;

    int screen_id;
};

class FluidSim : public atcg::Application
{
public:
    FluidSim(const atcg::WindowProps& props) : atcg::Application(props) { pushLayer(new FluidSimLayer("Layer")); }

    ~FluidSim() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.width  = 800;
    props.height = 800;
    return new FluidSim(props);
}