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
    data.sigma_a = dict.getValueOr<glm::vec3>("sigma_a", glm::vec3(0));
    data.sigma_s = dict.getValueOr<glm::vec3>("sigma_s", glm::vec3(0));
    data.Le      = glm::vec3(dict.getValueOr<glm::vec3>("Le", glm::vec3(0)));

    _data_buffer.upload(&data);
}

HomogeneousMedium::~HomogeneousMedium() {}

void PipelineInitializer<HomogeneousMedium>::apply(const atcg::ref_ptr<HomogeneousMedium>& component) const
{
    // TODO
    // if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    auto phase_function = component->getPhaseFunction();

    const std::string ptx_filename = "./bin/HomogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEvent"});

    uint32_t eval_transmittance_index =
        sbt->addCallableEntry(eval_transmittance_prog_group, component->getDataBuffer().get());
    uint32_t sample_medium_event_index =
        sbt->addCallableEntry(sample_medium_event_prog_group, component->getDataBuffer().get());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex   = eval_transmittance_index;
    vptr_table_data.sampleCallIndex = sample_medium_event_index;
    vptr_table_data.phase_function  = phase_function ? phase_function->getVPtrTable() : nullptr;

    component->getVPtrTableHolder().upload(&vptr_table_data);

    component->markInitialized();
}
}    // namespace atcg