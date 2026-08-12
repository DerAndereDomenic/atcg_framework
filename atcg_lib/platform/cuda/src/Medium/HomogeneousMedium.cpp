#include <Medium/HomogeneousMedium.h>

#include <Scene/ComponentRegistry.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

namespace atcg
{
HomogeneousMedium::HomogeneousMedium(const atcg::Dictionary& dict) : Medium(dict)
{
    HomogeneousMediumData data;
    data.albedo  = dict.getValueOr<glm::vec3>("albedo", glm::vec3(0));
    data.density = dict.getValueOr<float>("density", 0.0f);
    data.Le      = glm::vec3(dict.getValueOr<glm::vec3>("Le", glm::vec3(0)));

    _data_buffer.upload(&data);
}

HomogeneousMedium::~HomogeneousMedium() {}

void HomogeneousMedium::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                           const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    auto phase_function = getPhaseFunction();

    const std::string ptx_filename = "./bin/HomogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEvent"});

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