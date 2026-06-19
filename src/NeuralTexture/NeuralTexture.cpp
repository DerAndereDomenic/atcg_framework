#include "NeuralTexture.h"

#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <Utils/Utils.h>
#include <DataStructure/Timer.h>
#include <Core/Log.h>

torch::autograd::variable_list NeuralTextureNode::apply(torch::autograd::variable_list&& grads)
{
    auto grad_output = grads[0];

    auto grad_input = neural_texture->_backward(grad_output);
    return grad_input;
}

void NeuralTextureNode::release_variables()
{
    neural_texture = nullptr;
}

NeuralTexture::NeuralTexture(const atcg::ref_ptr<atcg::RaytracingContext>& context)
{
    _pipeline = atcg::make_ref<atcg::RayTracingPipeline>(context);
    _sbt      = atcg::make_ref<atcg::ShaderBindingTable>();

    const std::string ptx_raygen_filename = "./bin/NeuralTexture_ptx.ptx";
    OptixProgramGroup fwd_prog_group      = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__fwd"});
    OptixProgramGroup bckwd_prog_group    = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__bckwd"});

    _fwd_call_index   = _sbt->addRaygenEntry(fwd_prog_group);
    _bckwd_call_index = _sbt->addRaygenEntry(bckwd_prog_group);

    _pipeline->createPipeline();
    _sbt->createSBT();

    _weights = torch::empty({32 * 64 + 3 * 64 * 64 + 8 * 64}, atcg::TensorOptions::floatDeviceOptions());
    float a  = std::sqrt(6.0f / 128.0f);
    torch::nn::init::uniform_(_weights, -a, a);
    _weights = _weights.requires_grad_(true);
    _bias    = torch::zeros({64 + 64 + 64 + 64 + 8}, atcg::TensorOptions::floatDeviceOptions());
    _bias    = _bias.requires_grad_(true);

    _hash_grid    = atcg::HashGrid<half, 16, 2>(16, 512, (1 << 20));
    _hash_weights = _hash_grid.getWeights().to(torch::kFloat32).requires_grad_(true);

    _mlp = atcg::MLP<3, 32, 64, 8>(context, _weights.to(torch::kFloat16), _bias.to(torch::kFloat16));
}

NeuralTexture::~NeuralTexture() {}

torch::Tensor NeuralTexture::evaluate(const uint32_t width, const uint32_t height)
{
    const auto& parameters = getParameters();

    bool is_executable = parameters.size() > 0 && torch::autograd::GradMode::is_enabled() &&
                         torch::autograd::any_variable_requires_grad(parameters);

    torch::Tensor result;
    {
        torch::NoGradGuard no_grad;
        result = _forward(width, height);
    }

    if(is_executable)
    {
        std::shared_ptr<NeuralTextureNode> node(new NeuralTextureNode(), torch::autograd::deleteNode);
        auto next_edges = torch::autograd::collect_next_edges(parameters);
        node->set_next_edges(std::move(next_edges));
        node->neural_texture = this;

        torch::autograd::set_history(result, node);
    }

    return result;
}

torch::Tensor NeuralTexture::_forward(const uint32_t width, const uint32_t height)
{
    atcg::Timer timer;
    if(_weights_dirty)
    {
        _mlp.setWeights(_weights.to(torch::kFloat16));
        _mlp.setBias(_bias.to(torch::kFloat16));
        _mlp.uploadDeviceMLPData();

        _hash_grid.setWeights(_hash_weights.to(torch::kFloat16));
        _hash_grid.uploadDeviceHashGridData();

        _weights_dirty = false;
    }

    auto output_tensor = torch::zeros({height, width, 4}, atcg::TensorOptions::floatDeviceOptions());

    NeuralTextureData launch_params;
    launch_params.width            = width;
    launch_params.height           = height;
    launch_params.output           = (glm::vec4*)output_tensor.data_ptr();
    launch_params.device_mlp       = _mlp.getDeviceMLP();
    launch_params.device_hash_grid = _hash_grid.getDeviceHashGrid();

    _launch_params.upload(&launch_params);

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(NeuralTextureData),
                      _sbt->getSBT(_fwd_call_index),
                      width,
                      height,
                      1,
                      nullptr);


    ATCG_TRACE("NeuralTexture forward pass took {} ms", timer.elapsedMillis());
    return output_tensor;
}

std::vector<torch::Tensor> NeuralTexture::_backward(const torch::Tensor& grad_output)
{
    atcg::Timer timer;
    _weights_dirty = true;
    _mlp.zeroGradients();
    _hash_grid.zeroGradients();

    NeuralTextureData launch_params;
    launch_params.width            = grad_output.size(1);
    launch_params.height           = grad_output.size(0);
    launch_params.grad_output      = (glm::vec4*)grad_output.data_ptr();
    launch_params.device_mlp       = _mlp.getDeviceMLP();
    launch_params.device_hash_grid = _hash_grid.getDeviceHashGrid();

    _launch_params.upload(&launch_params);

    _pipeline->launch((CUdeviceptr)_launch_params.get(),
                      sizeof(NeuralTextureData),
                      _sbt->getSBT(_bckwd_call_index),
                      grad_output.size(1),
                      grad_output.size(0),
                      1,
                      nullptr);

    auto grad_weights = _mlp.getWeightGradients().to(torch::kFloat32);
    auto grad_bias    = _mlp.getBiasGradients().to(torch::kFloat32);
    auto grad_hash    = _hash_grid.getGradWeights().to(torch::kFloat32);

    grad_weights = torch::where(torch::isnan(grad_weights), torch::zeros_like(grad_weights), grad_weights);
    grad_bias    = torch::where(torch::isnan(grad_bias), torch::zeros_like(grad_bias), grad_bias);
    grad_hash    = torch::where(torch::isnan(grad_hash), torch::zeros_like(grad_hash), grad_hash);

    ATCG_TRACE("NeuralTexture backward pass took {} ms", timer.elapsedMillis());

    return {grad_weights, grad_bias, grad_hash};
}