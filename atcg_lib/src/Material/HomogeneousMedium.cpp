#include <Material/HomogeneousMedium.h>

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
HomogeneousMedium::HomogeneousMedium(const Dictionary& dict) : Medium("Homogeneous", dict)
{
    _albedo   = dict.getValueOr<glm::vec3>("albedo", _albedo);
    _density  = dict.getValueOr<float>("density", _density);
    _Le       = dict.getValueOr<float>("Le", _Le);
    _Le_color = dict.getValueOr<glm::vec3>("Le_color", _Le_color);

    _flags = MediumFlag::Homogeneous;
}

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

    shader->setVec3("albedo", _albedo);
    shader->setFloat("density", _density);
    shader->setFloat("Le", _Le);
    shader->setVec3("Le_color", _Le_color);

    _uploaded = true;
}

atcg::ref_ptr<Medium> HomogeneousMedium::clone() const
{
    auto medium       = atcg::make_ref<HomogeneousMedium>(atcg::Dictionary());
    medium->_albedo   = _albedo;
    medium->_density  = _density;
    medium->_Le       = _Le;
    medium->_Le_color = _Le_color;
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