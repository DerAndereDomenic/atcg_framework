#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>
#include <portable-file-dialogs.h>

#include <Core/Common.h>
#include <torch/optim.h>

#include "NeuralTexture.h"


class NeuralTextureLayer : public atcg::Layer
{
public:
    NeuralTextureLayer(const std::string& name) : atcg::Layer(name) {}

    // This is run at the start of the program
    virtual void onAttach() override
    {
        atcg::Application::get()->enableDockSpace(true);

        auto target_image = atcg::IO::imread("res/target.jpeg");
        _target_texture   = atcg::Texture2D::create(target_image);
        _target_tensor    = _target_texture->getData(atcg::GPU).to(torch::kFloat32) / 255.0f;

        // If not a 3 channel image, pad 4th channel with ones
        if(_target_tensor.size(2) == 3)
        {
            auto alpha_channel = torch::ones({_target_tensor.size(0), _target_tensor.size(1), 1},
                                             atcg::TensorOptions::floatDeviceOptions());
            _target_tensor     = torch::cat({_target_tensor, alpha_channel}, 2);
        }


        atcg::TextureSpecification spec;
        spec.width        = target_image->width();
        spec.height       = target_image->height();
        spec.format       = atcg::TextureFormat::RGBAFLOAT;
        _output_texture   = atcg::Texture2D::create(spec);
        auto optx_context = atcg::RaytracingContextManager::createContext();
        _neural_texture   = atcg::make_ref<NeuralTexture>(optx_context);

        optimizer =
            atcg::make_ref<torch::optim::Adam>(_neural_texture->getParameters(), torch::optim::AdamOptions(0.0005f));
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        optimizer->zero_grad();
        auto output = _neural_texture->evaluate(_output_texture->width(), _output_texture->height());
        _output_texture->setData(output);

        auto loss = 100.0f * torch::mean((output - _target_tensor) * (output - _target_tensor));

        try
        {
            loss.backward();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }


        optimizer->step();

        atcg::GraphicsCommand::beginRenderPass(atcg::Renderer::getFramebuffer());
        atcg::GraphicsCommand::clear();

        atcg::Renderer::drawImage(_output_texture);

        atcg::GraphicsCommand::endRenderPass();
    }

#ifndef ATCG_HEADLESS
    virtual void onImGuiRender() override {}
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(NeuralTextureLayer::onViewportResized));
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event) { return false; }


private:
    atcg::ref_ptr<atcg::Texture2D> _target_texture;
    torch::Tensor _target_tensor;
    atcg::ref_ptr<atcg::Texture2D> _output_texture;
    atcg::ref_ptr<NeuralTexture> _neural_texture;

    atcg::ref_ptr<torch::optim::Adam> optimizer;
};

class NeuralTextureApp : public atcg::Application
{
public:
    NeuralTextureApp(const atcg::WindowProps& props) : atcg::Application(props)
    {
        pushLayer(new NeuralTextureLayer("Layer"));
    }

    ~NeuralTextureApp() {}
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync  = true;
    props.width  = 1024;
    props.height = 1024;
    return new NeuralTextureApp(props);
}