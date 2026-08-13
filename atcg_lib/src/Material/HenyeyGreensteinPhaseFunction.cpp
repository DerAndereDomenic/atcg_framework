#include <Material/HenyeyGreensteinPhaseFunction.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

#define G_KEY    "g"
#define TYPE_KEY "Type"

namespace atcg
{
HenyeyGreensteinPhaseFunction::HenyeyGreensteinPhaseFunction(const Dictionary& dict)
    : PhaseFunction("HenyeyGreenstein", dict)
{
    _g = dict.getValueOr<float>("g", _g);
}

void HenyeyGreensteinPhaseFunction::uploadPhaseFunction(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader)
{
    shader->setFloat("g", _g);

    _uploaded = true;
}

void HenyeyGreensteinPhaseFunction::updateData()
{
    HenyeyGreensteinPhaseFunctionData data;
    data.g = _g;

    _phase_function_data_buffer.upload(&data);
}

void HenyeyGreensteinPhaseFunction::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/HenyeyGreensteinPhaseFunction_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_hgphase"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_hgphase"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group, _phase_function_data_buffer.get());
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group, _phase_function_data_buffer.get());

    PhaseFunctionVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;

    _phase_function_vptr_table.upload(&table);

    markInitialized();
}

atcg::ref_ptr<PhaseFunction> HenyeyGreensteinPhaseFunction::clone() const
{
    auto phase_function = atcg::make_ref<HenyeyGreensteinPhaseFunction>(atcg::Dictionary());
    phase_function->_g  = _g;
    return phase_function;
}

void PhaseFunctionSerializer<HenyeyGreensteinPhaseFunction>::serialize(
    const atcg::ref_ptr<HenyeyGreensteinPhaseFunction>& phase_function,
    const std::filesystem::path& path)
{
    nlohmann::json j;
    j["Version"] = "1.0";

    j[TYPE_KEY] = phase_function->getPhaseFunctionType();
    j[G_KEY]    = phase_function->g();

    std::ofstream o(path);
    o << std::setw(4) << j << std::endl;
}

atcg::ref_ptr<HenyeyGreensteinPhaseFunction>
PhaseFunctionSerializer<HenyeyGreensteinPhaseFunction>::deserialize(const std::filesystem::path& path,
                                                                    const nlohmann::json& phase_function_node)
{
    atcg::ref_ptr<HenyeyGreensteinPhaseFunction> phase_function =
        atcg::make_ref<HenyeyGreensteinPhaseFunction>(atcg::Dictionary());

    float g = phase_function_node.value(G_KEY, 0.0f);
    phase_function->setG(g);

    return phase_function;
}

bool PhaseFunctionGUIRenderer<HenyeyGreensteinPhaseFunction>::renderGUI(
    const atcg::ref_ptr<HenyeyGreensteinPhaseFunction>& phase_function,
    const std::string& key,
    bool& deactivated)
{
#ifndef ATCG_HEADLESS
    bool updated = false;


    float g = phase_function->g();
    if(ImGui::DragFloat((key + " g").c_str(), &g, 0.01f, -1.0f, 1.0f))
    {
        phase_function->setG(g);
        updated = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

#endif
    return updated;
}

void HenyeyGreensteinPhaseFunction::registerPhaseFunction(PhaseFunctionRegistry::Registry* registry)
{
    ATCG_REGISTER_PHASE_FUNCTION(registry, "HenyeyGreenstein", HenyeyGreensteinPhaseFunction);
}

}    // namespace atcg