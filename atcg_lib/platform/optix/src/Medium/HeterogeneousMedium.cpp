#include <Medium/HeterogeneousMedium.h>

#include <Renderer/Texture.h>

#include <Scene/ComponentRegistry.h>

namespace atcg
{

using GridComponent = HeterogeneousMediumComponent::GridComponent;

HeterogeneousMedium::HeterogeneousMedium(const Dictionary& dict) : Medium(dict)
{
    HeterogeneousMediumData data;

    glm::mat4 to_world       = dict.getValueOr<glm::mat4>("to_world", glm::mat4(1));
    glm::mat4 world_to_local = glm::inverse(to_world);

    auto density_grid     = dict.getValue<GridComponent>("density_grid");
    auto density_texture  = AssetManager::getAsset<Texture3D>(density_grid.handle);
    _density_tensor       = density_texture ? density_texture->getData(atcg::GPU) : torch::Tensor();
    auto emission_grid    = dict.getValue<GridComponent>("emission_grid");
    auto emission_texture = AssetManager::getAsset<Texture3D>(emission_grid.handle);
    _emission_tensor      = emission_texture ? emission_texture->getData(atcg::GPU) : torch::Tensor();
    auto albedo_grid      = dict.getValue<GridComponent>("albedo_grid");
    auto albedo_texture   = AssetManager::getAsset<Texture3D>(albedo_grid.handle);
    _albedo_tensor        = albedo_texture ? albedo_texture->getData(atcg::GPU) : torch::Tensor();

    auto density_majorant = _density_tensor.max().item<float>();

    data.density_grid.storage.sampler =
        TextureSampler<float>((std::byte*)_density_tensor.data_ptr(), density_texture->getSpecification());
    data.density_grid.scale = density_grid.scale;
    data.density_majorant   = density_majorant * data.density_grid.scale;
    {
        glm::mat4 to_uvw         = glm::mat4(1);
        glm::vec3 scale          = density_grid.bbox.max - density_grid.bbox.min;
        to_uvw                   = to_uvw * glm::scale(1.0f / scale);
        to_uvw                   = to_uvw * glm::translate(-density_grid.bbox.min);
        to_uvw                   = to_uvw * world_to_local;
        data.density_grid.to_uvw = to_uvw;
    }

    data.emission_grid.storage.sampler =
        emission_texture
            ? TextureSampler<glm::vec3>((std::byte*)_emission_tensor.data_ptr(), emission_texture->getSpecification())
            : TextureSampler<glm::vec3>();
    data.emission_grid.default_value = glm::vec3(0);
    data.emission_grid.scale         = emission_grid.scale;
    {
        glm::mat4 to_uvw          = glm::mat4(1);
        glm::vec3 scale           = emission_grid.bbox.max - emission_grid.bbox.min;
        to_uvw                    = to_uvw * glm::scale(1.0f / scale);
        to_uvw                    = to_uvw * glm::translate(-emission_grid.bbox.min);
        to_uvw                    = to_uvw * world_to_local;
        data.emission_grid.to_uvw = to_uvw;
    }

    data.albedo_grid.storage.sampler = albedo_texture ? TextureSampler<glm::vec3>((std::byte*)_albedo_tensor.data_ptr(),
                                                                                  albedo_texture->getSpecification())
                                                      : TextureSampler<glm::vec3>();
    data.albedo_grid.scale           = albedo_grid.scale;
    {
        glm::mat4 to_uvw        = glm::mat4(1);
        glm::vec3 scale         = albedo_grid.bbox.max - albedo_grid.bbox.min;
        to_uvw                  = to_uvw * glm::scale(1.0f / scale);
        to_uvw                  = to_uvw * glm::translate(-albedo_grid.bbox.min);
        to_uvw                  = to_uvw * world_to_local;
        data.albedo_grid.to_uvw = to_uvw;
    }

    _data_buffer.upload(&data);
}

HeterogeneousMedium::~HeterogeneousMedium() {}

void HeterogeneousMedium::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                             const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    auto phase_function = getPhaseFunction();

    const std::string ptx_filename = "./bin/HeterogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_sampleMediumEvent"});
    OptixProgramGroup sample_medium_event_backward_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_sampleMediumEventBackward"});
    OptixProgramGroup eval_transmittance_backward_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_evalTransmittanceBackward"});

    uint32_t eval_transmittance_index  = sbt->addCallableEntry(eval_transmittance_prog_group, _data_buffer.get());
    uint32_t sample_medium_event_index = sbt->addCallableEntry(sample_medium_event_prog_group, _data_buffer.get());
    uint32_t sample_medium_event_backward_index =
        sbt->addCallableEntry(sample_medium_event_backward_prog_group, _data_buffer.get());
    uint32_t eval_transmittance_backward_index =
        sbt->addCallableEntry(eval_transmittance_backward_prog_group, _data_buffer.get());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex                      = eval_transmittance_index;
    vptr_table_data.sampleCallIndex                    = sample_medium_event_index;
    vptr_table_data.sampleBackwardCallIndex            = sample_medium_event_backward_index;
    vptr_table_data.phase_function                     = phase_function ? phase_function->getVPtrTable() : nullptr;
    vptr_table_data.evalTransmittanceBackwardCallIndex = eval_transmittance_backward_index;

    _vptr_table.upload(&vptr_table_data);
    markInitialized();
}

void HeterogeneousMedium::onImGuiRender()
{
    if(ImGui::Button("Optimize Density"))
    {
        atcg::TextureSpecification spec;
        spec.width             = 256;
        spec.height            = 256;
        spec.depth             = 256;
        spec.format            = TextureFormat::RFLOAT;
        spec.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;

        _optimize_density = true;
        _optimizable      = true;

        _density_tensor = torch::ones({256, 256, 256}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        _density_grad_tensor = torch::zeros({256, 256, 256}, atcg::TensorOptions::floatDeviceOptions());

        HeterogeneousMediumData data;
        _data_buffer.download(&data);

        data.optimize_density             = true;
        data.density_majorant             = 1.0f;
        data.density_grid.storage.sampler = TextureSampler<float>((std::byte*)_density_tensor.data_ptr(), spec);
        data.density_grid.storage.writer  = TextureWriter<float>((std::byte*)_density_grad_tensor.data_ptr(), spec);

        _data_buffer.upload(&data);

        spec.depth            = 0;
        spec.format           = TextureFormat::RGFLOAT;
        _density_texture      = atcg::Texture2D::create(spec);
        _density_grad_texture = atcg::Texture2D::create(spec);
    }

    if(ImGui::Button("Optimize Albedo"))
    {
        atcg::TextureSpecification spec;
        spec.width             = 256;
        spec.height            = 256;
        spec.depth             = 256;
        spec.format            = TextureFormat::RGBFLOAT;
        spec.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;

        _optimize_albedo = true;
        _optimizable     = true;

        _albedo_tensor =
            torch::ones({256, 256, 256, 3}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        _albedo_grad_tensor = torch::zeros({256, 256, 256, 3}, atcg::TensorOptions::floatDeviceOptions());

        HeterogeneousMediumData data;
        _data_buffer.download(&data);

        data.optimize_albedo             = true;
        data.albedo_grid.storage.sampler = TextureSampler<glm::vec3>((std::byte*)_albedo_tensor.data_ptr(), spec);
        data.albedo_grid.storage.writer  = TextureWriter<glm::vec3>((std::byte*)_albedo_grad_tensor.data_ptr(), spec);
        data.albedo_grid.scale           = 1.0f;

        _data_buffer.upload(&data);

        spec.depth           = 0;
        spec.format          = TextureFormat::RGBFLOAT;
        _albedo_texture      = atcg::Texture2D::create(spec);
        _albedo_grad_texture = atcg::Texture2D::create(spec);
    }

    auto normalize = [](torch::Tensor inp) -> torch::Tensor
    {
        auto min = torch::amin(inp);
        auto max = torch::amax(inp);
        auto y   = (inp - min) / (max - min);

        return y;
    };

    auto pos_neg = [normalize](torch::Tensor inp) -> torch::Tensor
    {
        torch::Tensor pos = torch::relu(inp);

        torch::Tensor neg = torch::relu(-inp);

        torch::Tensor y = torch::concat({pos, neg}, /*dim=*/-1);

        return normalize(y);
    };

    if(_optimize_density)
    {
        ImGui::SliderInt("Layer##density", &_layer_density, 0, 255);

        if(_density_tensor.defined())
        {
            auto density_slice =
                _density_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), _layer_density})
                    .unsqueeze(-1)
                    .contiguous();
            _density_texture->setData(pos_neg(density_slice));
            ImGui::Image((ImTextureID)_density_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }

        if(_density_tensor.grad().defined())
        {
            auto density_grad_slice = _density_tensor.grad()
                                          .index({torch::indexing::Slice(), torch::indexing::Slice(), _layer_density})
                                          .unsqueeze(-1)
                                          .contiguous();

            _density_grad_texture->setData(pos_neg(density_grad_slice));

            ImGui::Image((ImTextureID)_density_grad_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }
    }

    if(_optimize_albedo)
    {
        ImGui::SliderInt("Layer##albedo", &_layer_albedo, 0, 255);

        if(_albedo_tensor.defined())
        {
            auto albedo_slice =
                _albedo_tensor
                    .index(
                        {torch::indexing::Slice(), torch::indexing::Slice(), _layer_albedo, torch::indexing::Slice()})
                    .contiguous();
            _albedo_texture->setData(albedo_slice);
            ImGui::Image((ImTextureID)_albedo_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }

        if(_albedo_tensor.grad().defined())
        {
            auto albedo_grad_slice =
                _albedo_tensor.grad()
                    .index(
                        {torch::indexing::Slice(), torch::indexing::Slice(), _layer_albedo, torch::indexing::Slice()})
                    .contiguous();

            _albedo_grad_texture->setData(albedo_grad_slice);

            ImGui::Image((ImTextureID)_albedo_grad_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }
    }
}

std::vector<torch::Tensor> HeterogeneousMedium::getParameters() const
{
    std::vector<torch::Tensor> params;

    if(_optimize_albedo) params.push_back(_albedo_tensor);
    if(_optimize_density) params.push_back(_density_tensor);

    return params;
}

std::vector<torch::Tensor> HeterogeneousMedium::getParameterGradients() const
{
    std::vector<torch::Tensor> gradients;

    if(_optimize_albedo) gradients.push_back(_albedo_grad_tensor);
    if(_optimize_density) gradients.push_back(_density_grad_tensor);

    return gradients;
}

void HeterogeneousMedium::zeroGrad()
{
    if(_optimize_albedo) _albedo_grad_tensor.zero_();
    if(_optimize_density) _density_grad_tensor.zero_();
}

void HeterogeneousMedium::HeterogeneousMedium::markOptimizable()
{
    // TODO
    // _optimize_albedo  = true;
    // _optimize_density = true;
    // _optimizable      = true;

    atcg::TextureSpecification spec;
    spec.width             = 256;
    spec.height            = 256;
    spec.depth             = 256;
    spec.format            = TextureFormat::RFLOAT;
    spec.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;

    _optimize_density = true;
    _optimizable      = true;

    _density_tensor      = torch::ones({256, 256, 256}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
    _density_grad_tensor = torch::zeros({256, 256, 256}, atcg::TensorOptions::floatDeviceOptions());

    HeterogeneousMediumData data;
    _data_buffer.download(&data);

    data.optimize_density             = true;
    data.density_majorant             = 1.0f;
    data.density_grid.storage.sampler = TextureSampler<float>((std::byte*)_density_tensor.data_ptr(), spec);
    data.density_grid.storage.writer  = TextureWriter<float>((std::byte*)_density_grad_tensor.data_ptr(), spec);
    data.density_grid.scale           = 1.0f;

    _data_buffer.upload(&data);

    spec.depth            = 0;
    spec.format           = TextureFormat::RGFLOAT;
    _density_texture      = atcg::Texture2D::create(spec);
    _density_grad_texture = atcg::Texture2D::create(spec);
}

void HeterogeneousMedium::clampParameters()
{
    if(_optimize_density)
    {
        // Clamp density to be non-negative.
        _density_tensor.clamp_(0.0f);
        _albedo_tensor.clamp_(0.0f, 1.0f);

        HeterogeneousMediumData data;
        _data_buffer.download(&data);
        data.density_majorant = _density_tensor.max().item<float>();
        _data_buffer.upload(&data);
    }
}
}    // namespace atcg