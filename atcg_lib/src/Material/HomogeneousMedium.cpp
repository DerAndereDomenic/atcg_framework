#include <Material/HomogeneousMedium.h>

#include <Material/HomogeneousMediumData.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

#define ALBEDO_KEY   "albedo"
#define DENSITY_KEY  "density"
#define G_KEY        "g"
#define LE_KEY       "Le"
#define LE_COLOR_KEY "Le_color"
#define TYPE_KEY     "Type"

namespace atcg
{

class HomogeneousMedium::Impl
{
public:
    Impl(const atcg::Dictionary& dict);

    ~Impl();

    glm::vec3 _albedo   = glm::vec3(0);
    float _density      = 0.0f;
    float _Le           = 0.0f;
    glm::vec3 _Le_color = glm::vec3(1);

    // GPU interface
    atcg::dref_ptr<HomogeneousMediumData> _medium_data_buffer;
};

HomogeneousMedium::Impl::Impl(const atcg::Dictionary& dict)
{
    _albedo   = dict.getValueOr<glm::vec3>("albedo", _albedo);
    _density  = dict.getValueOr<float>("density", _density);
    _Le       = dict.getValueOr<float>("Le", _Le);
    _Le_color = dict.getValueOr<glm::vec3>("Le_color", _Le_color);
}

HomogeneousMedium::Impl::~Impl() {}

HomogeneousMedium::HomogeneousMedium(const Dictionary& dict) : Medium("Homogeneous", dict)
{
    impl = std::make_unique<Impl>(dict);

    _flags = MediumFlag::Homogeneous;
}

HomogeneousMedium::~HomogeneousMedium() {}

void HomogeneousMedium::uploadMedium(RendererSystem* renderer,
                                     const atcg::ref_ptr<Shader>& shader,
                                     const glm::mat4& model)
{
    // TODO: Not used here but needs to be set
    uint32_t albedo_id   = renderer->popTextureID();
    _texture_ids[0]      = albedo_id;
    uint32_t density_id  = renderer->popTextureID();
    _texture_ids[1]      = density_id;
    uint32_t emission_id = renderer->popTextureID();
    _texture_ids[2]      = emission_id;

    shader->setVec3("albedo", impl->_albedo);
    shader->setFloat("density", impl->_density);
    shader->setFloat("Le", impl->_Le);
    shader->setVec3("Le_color", impl->_Le_color);

    _uploaded = true;
}

void HomogeneousMedium::setAlbedo(const glm::vec3& albedo)
{
    impl->_albedo = albedo;
}

void HomogeneousMedium::setDensity(const float density)
{
    impl->_density = density;
}

void HomogeneousMedium::setLe(const float Le)
{
    impl->_Le = Le;
}

void HomogeneousMedium::setLeColor(const glm::vec3& Le_color)
{
    impl->_Le_color = Le_color;
}

glm::vec3 HomogeneousMedium::albedo() const
{
    return impl->_albedo;
}

float HomogeneousMedium::density() const
{
    return impl->_density;
}

float HomogeneousMedium::Le() const
{
    return impl->_Le;
}

glm::vec3 HomogeneousMedium::Le_color() const
{
    return impl->_Le_color;
}


void HomogeneousMedium::updateData()
{
    HomogeneousMediumData data;
    data.albedo  = impl->_albedo;
    data.density = impl->_density;
    data.Le      = impl->_Le * impl->_Le_color;

    impl->_medium_data_buffer.upload(&data);
}

void HomogeneousMedium::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                           const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    updateData();
    if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    const std::string ptx_filename = "./bin/HomogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__homogeneousMedium_sampleMediumEvent"});

    uint32_t eval_transmittance_index =
        sbt->addCallableEntry(eval_transmittance_prog_group, impl->_medium_data_buffer.get());
    uint32_t sample_medium_event_index =
        sbt->addCallableEntry(sample_medium_event_prog_group, impl->_medium_data_buffer.get());

    MediumVPtrTable vptr_table_data;
    vptr_table_data.evalCallIndex   = eval_transmittance_index;
    vptr_table_data.sampleCallIndex = sample_medium_event_index;
    vptr_table_data.phase_function  = _phase_function ? _phase_function->getVPtrTable() : nullptr;

    _medium_vptr_table.upload(&vptr_table_data);

    markInitialized();
}

atcg::ref_ptr<Medium> HomogeneousMedium::clone() const
{
    auto medium = atcg::make_ref<HomogeneousMedium>(atcg::Dictionary());
    medium->setAlbedo(impl->_albedo);
    medium->setDensity(impl->_density);
    medium->setLe(impl->_Le);
    medium->setLeColor(impl->_Le_color);
    return medium;
}

void MediumSerializer<HomogeneousMedium>::serialize(const atcg::ref_ptr<HomogeneousMedium>& medium,
                                                    const std::filesystem::path& path)
{
    nlohmann::json j;
    j["Version"] = "1.0";

    j[TYPE_KEY]      = medium->getMediumType();
    glm::vec3 albedo = medium->albedo();
    float density    = medium->density();
    // float g            = medium->g(); TODO
    float Le           = medium->Le();
    glm::vec3 Le_color = medium->Le_color();

    j[ALBEDO_KEY]  = nlohmann::json::array({albedo.x, albedo.y, albedo.z});
    j[DENSITY_KEY] = density;
    // j[G_KEY]        = g; TODO
    j[LE_KEY]       = Le;
    j[LE_COLOR_KEY] = nlohmann::json::array({Le_color.x, Le_color.y, Le_color.z});

    std::ofstream o(path);
    o << std::setw(4) << j << std::endl;
}

atcg::ref_ptr<HomogeneousMedium> MediumSerializer<HomogeneousMedium>::deserialize(const std::filesystem::path& path,
                                                                                  const nlohmann::json& medium_node)
{
    atcg::ref_ptr<HomogeneousMedium> material = atcg::make_ref<HomogeneousMedium>(atcg::Dictionary());

    std::vector<float> albedo   = medium_node.value(ALBEDO_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    std::vector<float> Le_color = medium_node.value(LE_COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});

    material->setAlbedo(glm::make_vec3(albedo.data()));
    // material->setG(medium_node[G_KEY]); TOOD
    material->setLe(medium_node[LE_KEY]);
    material->setLeColor(glm::make_vec3(Le_color.data()));
    material->setDensity(medium_node[DENSITY_KEY]);
    return material;
}

bool MediumGUIRenderer<HomogeneousMedium>::renderGUI(const atcg::ref_ptr<HomogeneousMedium>& medium,
                                                     const std::string& key,
                                                     bool& deactivated)
{
#ifndef ATCG_HEADLESS
    bool updated     = false;
    glm::vec3 albedo = medium->albedo();
    if(ImGui::ColorEdit3((key + " Albedo").c_str(), glm::value_ptr(albedo)))
    {
        medium->setAlbedo(albedo);
        updated = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    float density = medium->density();
    if(ImGui::DragFloat((key + " Density").c_str(), &density, 0.05f, 0.0f, 50.0f))
    {
        medium->setDensity(density);
        updated = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    float Le = medium->Le();
    if(ImGui::DragFloat((key + " Le").c_str(), &Le, 0.01f, 0.0f, 100.0f))
    {
        medium->setLe(Le);
        updated = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    glm::vec3 Le_color = medium->Le_color();
    if(ImGui::ColorEdit3((key + " Le Color").c_str(), glm::value_ptr(Le_color)))
    {
        medium->setLeColor(Le_color);
        updated = true;
    }
    deactivated = ImGui::IsItemDeactivated() || deactivated;

#endif
    return updated;
}

void HomogeneousMedium::registerMedium(MediumRegistry::Registry* registry)
{
    ATCG_REGISTER_MEDIUM(registry, "Homogeneous", HomogeneousMedium);
}

}    // namespace atcg