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
    auto density_texture  = AssetManager::getAsset<Texture3D>(density_grid.handle)->clone();
    _density_texture      = std::static_pointer_cast<Texture3D>(density_texture);
    auto emission_grid    = dict.getValue<GridComponent>("emission_grid");
    auto emission_texture = AssetManager::getAsset<Texture3D>(emission_grid.handle);
    _emission_texture     = emission_texture ? std::static_pointer_cast<Texture3D>(emission_texture->clone()) : nullptr;
    auto albedo_grid      = dict.getValue<GridComponent>("albedo_grid");
    auto albedo_texture   = AssetManager::getAsset<Texture3D>(albedo_grid.handle);
    _albedo_texture       = albedo_texture ? std::static_pointer_cast<Texture3D>(albedo_texture->clone()) : nullptr;

    auto density_majorant = _density_texture->getData(atcg::GPU).max().item<float>();

    data.density_grid.storage.texture = _density_texture->getTextureObject();
    data.density_grid.scale           = density_grid.scale;
    data.density_majorant             = density_majorant * data.density_grid.scale;
    {
        glm::mat4 to_uvw         = glm::mat4(1);
        glm::vec3 scale          = density_grid.bbox.max - density_grid.bbox.min;
        to_uvw                   = to_uvw * glm::scale(1.0f / scale);
        to_uvw                   = to_uvw * glm::translate(-density_grid.bbox.min);
        to_uvw                   = to_uvw * world_to_local;
        data.density_grid.to_uvw = to_uvw;
    }

    data.emission_grid.storage.texture = _emission_texture ? _emission_texture->getTextureObject() : 0;
    data.emission_grid.default_value   = glm::vec3(0);
    data.emission_grid.scale           = emission_grid.scale;
    {
        glm::mat4 to_uvw          = glm::mat4(1);
        glm::vec3 scale           = emission_grid.bbox.max - emission_grid.bbox.min;
        to_uvw                    = to_uvw * glm::scale(1.0f / scale);
        to_uvw                    = to_uvw * glm::translate(-emission_grid.bbox.min);
        to_uvw                    = to_uvw * world_to_local;
        data.emission_grid.to_uvw = to_uvw;
    }

    data.albedo_grid.storage.texture = _albedo_texture ? _albedo_texture->getTextureObject() : 0;
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

HeterogeneousMedium::~HeterogeneousMedium()
{
    _density_texture->unmapDevicePointers();
    if(_albedo_texture) _albedo_texture->unmapDevicePointers();
    if(_emission_texture) _emission_texture->unmapDevicePointers();
}

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

    uint32_t eval_transmittance_index  = sbt->addCallableEntry(eval_transmittance_prog_group, _data_buffer.get());
    uint32_t sample_medium_event_index = sbt->addCallableEntry(sample_medium_event_prog_group, _data_buffer.get());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex   = eval_transmittance_index;
    vptr_table_data.sampleCallIndex = sample_medium_event_index;
    vptr_table_data.phase_function  = phase_function ? phase_function->getVPtrTable() : nullptr;

    _vptr_table.upload(&vptr_table_data);
    markInitialized();
}
}    // namespace atcg