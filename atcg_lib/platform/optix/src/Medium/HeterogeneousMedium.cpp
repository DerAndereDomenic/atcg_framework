#include <Medium/HeterogeneousMedium.h>

#include <Renderer/Texture.h>

#include <Scene/ComponentRegistry.h>
#include <Scene/Components/HeterogeneousMediumComponent.h>

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
    auto density_tensor   = density_texture ? density_texture->getData(atcg::GPU) : torch::Tensor();
    auto emission_grid    = dict.getValue<GridComponent>("emission_grid");
    auto emission_texture = AssetManager::getAsset<Texture3D>(emission_grid.handle);
    _emission_tensor      = emission_texture ? emission_texture->getData(atcg::GPU) : torch::Tensor();
    auto albedo_grid      = dict.getValue<GridComponent>("albedo_grid");
    auto albedo_texture   = AssetManager::getAsset<Texture3D>(albedo_grid.handle);
    auto albedo_tensor    = albedo_texture ? albedo_texture->getData(atcg::GPU) : torch::Tensor();

    auto density_majorant = density_tensor.max().item<float>();

    data.density_grid.storage.sampler =
        TextureSampler<float>((std::byte*)density_tensor.data_ptr(), density_texture->getSpecification());
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

    data.albedo_grid.storage.sampler = albedo_texture ? TextureSampler<glm::vec3>((std::byte*)albedo_tensor.data_ptr(),
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

    setParameter("density", density_tensor);
    setParameter("albedo", albedo_tensor);

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
        markParameterAsOptimizable("density");
    }

    if(ImGui::Button("Optimize Albedo"))
    {
        markParameterAsOptimizable("albedo");
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

    if(isParameterOptimizable("density"))
    {
        ImGui::SliderInt("Layer##density", &_layer_density, 0, 255);

        auto density_tensor = getParameter("density");
        if(density_tensor.defined())
        {
            auto density_slice =
                density_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(), _layer_density})
                    .unsqueeze(-1)
                    .contiguous();
            _density_texture->setData(pos_neg(density_slice));
            ImGui::Image((ImTextureID)_density_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }

        if(density_tensor.grad().defined())
        {
            auto density_grad_slice = density_tensor.grad()
                                          .index({torch::indexing::Slice(), torch::indexing::Slice(), _layer_density})
                                          .unsqueeze(-1)
                                          .contiguous();

            _density_grad_texture->setData(pos_neg(density_grad_slice));

            ImGui::Image((ImTextureID)_density_grad_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }
    }

    if(isParameterOptimizable("albedo"))
    {
        ImGui::SliderInt("Layer##albedo", &_layer_albedo, 0, 255);

        auto albedo_tensor = getParameter("albedo");
        if(albedo_tensor.defined())
        {
            auto albedo_slice =
                albedo_tensor
                    .index(
                        {torch::indexing::Slice(), torch::indexing::Slice(), _layer_albedo, torch::indexing::Slice()})
                    .contiguous();
            _albedo_texture->setData(albedo_slice);
            ImGui::Image((ImTextureID)_albedo_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }

        if(albedo_tensor.grad().defined())
        {
            auto albedo_grad_slice =
                albedo_tensor.grad()
                    .index(
                        {torch::indexing::Slice(), torch::indexing::Slice(), _layer_albedo, torch::indexing::Slice()})
                    .contiguous();

            _albedo_grad_texture->setData(albedo_grad_slice);

            ImGui::Image((ImTextureID)_albedo_grad_texture->getID(), ImVec2(256, 256), ImVec2 {0, 1}, ImVec2 {1, 0});
        }
    }
}

void HeterogeneousMedium::clampParameters()
{
    if(isParameterOptimizable("density"))
    {
        auto density_tensor = getParameter("density");
        // Clamp density to be non-negative.
        density_tensor.clamp_(0.0f);
    }
    else if(isParameterOptimizable("albedo"))
    {
        auto albedo_tensor = getParameter("albedo");
        albedo_tensor.clamp_(0.0f, 1.0f);
    }
}

void HeterogeneousMedium::markParameterAsOptimizable(const std::string& parameter_name)
{
    if(parameter_name == "density")
    {
        atcg::TextureSpecification spec;
        spec.width             = 256;
        spec.height            = 256;
        spec.depth             = 256;
        spec.format            = TextureFormat::RFLOAT;
        spec.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;

        auto density_tensor =
            torch::ones({256, 256, 256}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        setParameter("density", density_tensor);
        auto density_grad_tensor = getGradient("density");

        HeterogeneousMediumData data;
        _data_buffer.download(&data);

        data.optimize_density             = true;
        data.density_majorant             = 1.0f;
        data.density_grid.storage.sampler = TextureSampler<float>((std::byte*)density_tensor.data_ptr(), spec);
        data.density_grid.storage.writer  = TextureWriter<float>((std::byte*)density_grad_tensor.data_ptr(), spec);

        _data_buffer.upload(&data);

        spec.depth            = 0;
        spec.format           = TextureFormat::RGFLOAT;
        _density_texture      = atcg::Texture2D::create(spec);
        _density_grad_texture = atcg::Texture2D::create(spec);
    }
    else if(parameter_name == "albedo")
    {
        atcg::TextureSpecification spec;
        spec.width             = 256;
        spec.height            = 256;
        spec.depth             = 256;
        spec.format            = TextureFormat::RGBFLOAT;
        spec.sampler.wrap_mode = TextureWrapMode::CLAMP_TO_EDGE;

        auto albedo_tensor =
            torch::ones({256, 256, 256, 3}, atcg::TensorOptions::floatDeviceOptions()).requires_grad_(true);
        setParameter("albedo", albedo_tensor);
        auto albedo_grad_tensor = getGradient("albedo");

        HeterogeneousMediumData data;
        _data_buffer.download(&data);

        data.optimize_albedo             = true;
        data.albedo_grid.storage.sampler = TextureSampler<glm::vec3>((std::byte*)albedo_tensor.data_ptr(), spec);
        data.albedo_grid.storage.writer  = TextureWriter<glm::vec3>((std::byte*)albedo_grad_tensor.data_ptr(), spec);
        data.albedo_grid.scale           = 1.0f;

        _data_buffer.upload(&data);

        spec.depth           = 0;
        spec.format          = TextureFormat::RGBFLOAT;
        _albedo_texture      = atcg::Texture2D::create(spec);
        _albedo_grad_texture = atcg::Texture2D::create(spec);
    }
}

}    // namespace atcg