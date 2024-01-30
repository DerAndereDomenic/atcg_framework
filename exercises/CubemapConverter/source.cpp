#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <ImGuizmo.h>

#include <random>
class ConverterLayer : public atcg::Layer
{
public:
    ConverterLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(false);

        scene = atcg::make_ref<atcg::Scene>();

        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = atcg::make_ref<atcg::FirstPersonController>(aspect_ratio);

        auto face_0                         = atcg::IO::imread("../Cubemap/image_2.png");
        auto face_1                         = atcg::IO::imread("../Cubemap/image_0.png");
        auto face_2                         = atcg::IO::imread("../Cubemap/image_5.png");
        auto face_3                         = atcg::IO::imread("../Cubemap/image_4.png");
        auto face_4                         = atcg::IO::imread("../Cubemap/image_1.png");
        auto face_5                         = atcg::IO::imread("../Cubemap/image_3.png");
        atcg::ref_ptr<atcg::Image> images[] = {face_0, face_1, face_2, face_3, face_4, face_5};

        bool is_hdr  = face_0->isHDR();
        int channels = face_0->channels();
        int width    = face_0->width();
        int height   = face_0->height();

        torch::Tensor cubemap_tensor =
            torch::empty({6, height, width, channels},
                         is_hdr ? atcg::TensorOptions::floatHostOptions() : atcg::TensorOptions::uint8HostOptions());

        for(int i = 0; i < 6; ++i)
        {
            cubemap_tensor.index_put_({i, torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice()},
                                      atcg::createTensorFromPointer((void*)images[i]->data(),
                                                                    {height, width, channels},
                                                                    is_hdr ? atcg::TensorOptions::floatHostOptions()
                                                                           : atcg::TensorOptions::uint8HostOptions()));
        }

        atcg::TextureSpecification spec;
        spec.width  = width;
        spec.height = height;
        switch(channels)
        {
            case 1:
            {
                spec.format = (is_hdr ? atcg::TextureFormat::RFLOAT : atcg::TextureFormat::RINT8);
            }
            break;
            case 2:
            {
                spec.format = (is_hdr ? atcg::TextureFormat::RGFLOAT : atcg::TextureFormat::RG);
            }
            break;
            case 3:
            {
                spec.format = (is_hdr ? atcg::TextureFormat::RGBFLOAT : atcg::TextureFormat::RGB);
            }
            break;
            case 4:
            {
                spec.format = (is_hdr ? atcg::TextureFormat::RGBAFLOAT : atcg::TextureFormat::RGBA);
            }
            break;
        }

        cubemap = atcg::TextureCube::create(spec);
        cubemap->setData(cubemap_tensor);

        std::vector<atcg::Vertex> vertices;
        vertices.push_back(atcg::Vertex(glm::vec3(-1, -1, 0)));
        vertices.push_back(atcg::Vertex(glm::vec3(1, -1, 0)));
        vertices.push_back(atcg::Vertex(glm::vec3(1, 1, 0)));
        vertices.push_back(atcg::Vertex(glm::vec3(-1, 1, 0)));

        std::vector<glm::u32vec3> indices;
        indices.push_back(glm::u32vec3(0, 1, 2));
        indices.push_back(glm::u32vec3(0, 2, 3));

        auto quad = atcg::Graph::createTriangleMesh(vertices, indices);

        spec.width          = 4096;
        spec.height         = 2048;
        spec.format         = atcg::TextureFormat::RGBAFLOAT;
        auto skybox_texture = atcg::Texture2D::create(spec);

        auto frame_buffer = atcg::make_ref<atcg::Framebuffer>(spec.width, spec.height);
        frame_buffer->attachTexture(skybox_texture);
        frame_buffer->complete();

        auto converter_shader = atcg::make_ref<atcg::Shader>("exercises/CubemapConverter/converter.vs",
                                                             "exercises/CubemapConverter/converter.fs");

        frame_buffer->use();
        atcg::Renderer::setViewport(0, 0, 4096, 2048);
        atcg::Renderer::draw(quad, {}, glm::mat4(1), glm::vec3(1), converter_shader);
        atcg::Framebuffer::useDefault();
        atcg::Renderer::setViewport(0, 0, window->getWidth(), window->getHeight());

        auto texture_data = skybox_texture->getData(atcg::CPU);

        atcg::ref_ptr<atcg::Image> output_img =
            atcg::make_ref<atcg::Image>((const float*)texture_data.data_ptr(), spec.width, spec.height, 4);

        output_img->store("skybox.hdr");

        atcg::Renderer::setSkybox(output_img);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        atcg::Renderer::draw(scene, camera_controller->getCamera());
    }

    virtual void onImGuiRender() override {}

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override { camera_controller->onEvent(event); }

private:
    atcg::ref_ptr<atcg::Scene> scene;

    atcg::ref_ptr<atcg::TextureCube> cubemap;

    atcg::ref_ptr<atcg::FirstPersonController> camera_controller;
};

class Converter : public atcg::Application
{
public:
    Converter() : atcg::Application() { pushLayer(new ConverterLayer("Layer")); }

    ~Converter() {}
};

atcg::Application* atcg::createApplication()
{
    return new Converter;
}