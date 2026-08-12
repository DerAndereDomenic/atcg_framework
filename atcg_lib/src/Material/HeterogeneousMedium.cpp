#include <Material/HeterogeneousMedium.h>

#include <Asset/AssetManagerSystem.h>
#include <Utils/Utils.h>

#ifndef ATCG_HEADLESS
    #include <imgui.h>
#endif

#define DENSITY_GRID_KEY  "density_grid"
#define ALBEDO_GRID_KEY   "albedo_grid"
#define EMISSION_GRID_KEY "emission_grid"
#define G_KEY             "g"
#define GRID_KEY          "grid"
#define SCALE_KEY         "scale"
#define BBOX_KEY          "bbox"
#define BBOX_MIN_KEY      "min"
#define BBOX_MAX_KEY      "max"
#define TYPE_KEY          "Type"

namespace atcg
{
HeterogeneousMedium::HeterogeneousMedium(const Dictionary& dict) : Medium("Heterogeneous", dict)
{
    TextureSpecification spec;
    spec.width  = 1;
    spec.height = 1;
    spec.depth  = 1;
    spec.format = TextureFormat::RGBFLOAT;
    glm::vec3 black(0);
    glm::vec3 white(1.0f);
    _default_emission_texture = atcg::Texture3D::create(&black, spec);

    _default_albedo_texture = atcg::Texture3D::create(&white, spec);
}

atcg::ref_ptr<Texture3D> HeterogeneousMedium::density() const
{
    return AssetManager::getAsset<Texture3D>(density_grid.handle);
}

atcg::ref_ptr<Texture3D> HeterogeneousMedium::albedo() const
{
    return AssetManager::isAssetHandleValid(albedo_grid.handle) ? AssetManager::getAsset<Texture3D>(albedo_grid.handle)
                                                                : _default_albedo_texture;
}

atcg::ref_ptr<Texture3D> HeterogeneousMedium::emission() const
{
    return AssetManager::isAssetHandleValid(emission_grid.handle)
               ? AssetManager::getAsset<Texture3D>(emission_grid.handle)
               : _default_emission_texture;
}

atcg::ref_ptr<Medium> HeterogeneousMedium::clone() const
{
    auto medium = atcg::make_ref<HeterogeneousMedium>(atcg::Dictionary());

    medium->density_grid  = density_grid;
    medium->albedo_grid   = albedo_grid;
    medium->emission_grid = emission_grid;

    return medium;
}

void MediumSerializer<HeterogeneousMedium>::serialize(const atcg::ref_ptr<HeterogeneousMedium>& medium,
                                                      const std::filesystem::path& path)
{
    nlohmann::json j;
    j["Version"] = "1.0";

    nlohmann::json j_density_grid;
    nlohmann::json j_albedo_grid;
    nlohmann::json j_emission_grid;

    j_density_grid[GRID_KEY]  = (uint64_t)medium->density_grid.handle;
    j_density_grid[SCALE_KEY] = medium->density_grid.scale;
    nlohmann::json j_bbox_density;
    j_bbox_density[BBOX_MIN_KEY] = nlohmann::json::array(
        {medium->density_grid.bbox.min.x, medium->density_grid.bbox.min.y, medium->density_grid.bbox.min.z});
    j_bbox_density[BBOX_MAX_KEY] = nlohmann::json::array(
        {medium->density_grid.bbox.max.x, medium->density_grid.bbox.max.y, medium->density_grid.bbox.max.z});
    j_density_grid[BBOX_KEY] = j_bbox_density;

    j_albedo_grid[GRID_KEY]  = (uint64_t)medium->albedo_grid.handle;
    j_albedo_grid[SCALE_KEY] = medium->albedo_grid.scale;
    nlohmann::json j_bbox_albedo;
    j_bbox_albedo[BBOX_MIN_KEY] = nlohmann::json::array(
        {medium->albedo_grid.bbox.min.x, medium->albedo_grid.bbox.min.y, medium->albedo_grid.bbox.min.z});
    j_bbox_albedo[BBOX_MAX_KEY] = nlohmann::json::array(
        {medium->albedo_grid.bbox.max.x, medium->albedo_grid.bbox.max.y, medium->albedo_grid.bbox.max.z});
    j_albedo_grid[BBOX_KEY] = j_bbox_albedo;

    j_emission_grid[GRID_KEY]  = (uint64_t)medium->emission_grid.handle;
    j_emission_grid[SCALE_KEY] = medium->emission_grid.scale;
    nlohmann::json j_bbox_emission;
    j_bbox_emission[BBOX_MIN_KEY] = nlohmann::json::array(
        {medium->emission_grid.bbox.min.x, medium->emission_grid.bbox.min.y, medium->emission_grid.bbox.min.z});
    j_bbox_emission[BBOX_MAX_KEY] = nlohmann::json::array(
        {medium->emission_grid.bbox.max.x, medium->emission_grid.bbox.max.y, medium->emission_grid.bbox.max.z});
    j_emission_grid[BBOX_KEY] = j_bbox_emission;

    j[TYPE_KEY]          = medium->getMediumType();
    j[DENSITY_GRID_KEY]  = j_density_grid;
    j[ALBEDO_GRID_KEY]   = j_albedo_grid;
    j[EMISSION_GRID_KEY] = j_emission_grid;
    // j[G_KEY]             = medium->g;

    std::ofstream o(path);
    o << std::setw(4) << j << std::endl;
}

atcg::ref_ptr<HeterogeneousMedium> MediumSerializer<HeterogeneousMedium>::deserialize(const std::filesystem::path& path,
                                                                                      const nlohmann::json& medium_node)
{
    atcg::ref_ptr<HeterogeneousMedium> material = atcg::make_ref<HeterogeneousMedium>(atcg::Dictionary());

    // --- Density grid ---
    const auto& j_density         = medium_node[DENSITY_GRID_KEY];
    material->density_grid.handle = (AssetHandle)j_density[GRID_KEY].get<uint64_t>();
    material->density_grid.scale  = j_density[SCALE_KEY].get<float>();

    const auto& j_bbox_density      = j_density[BBOX_KEY];
    material->density_grid.bbox.min = glm::vec3(j_bbox_density[BBOX_MIN_KEY][0].get<float>(),
                                                j_bbox_density[BBOX_MIN_KEY][1].get<float>(),
                                                j_bbox_density[BBOX_MIN_KEY][2].get<float>());
    material->density_grid.bbox.max = glm::vec3(j_bbox_density[BBOX_MAX_KEY][0].get<float>(),
                                                j_bbox_density[BBOX_MAX_KEY][1].get<float>(),
                                                j_bbox_density[BBOX_MAX_KEY][2].get<float>());

    // --- Albedo grid ---
    const auto& j_albedo         = medium_node[ALBEDO_GRID_KEY];
    material->albedo_grid.handle = (AssetHandle)j_albedo[GRID_KEY].get<uint64_t>();
    material->albedo_grid.scale  = j_albedo[SCALE_KEY].get<float>();

    const auto& j_bbox_albedo      = j_albedo[BBOX_KEY];
    material->albedo_grid.bbox.min = glm::vec3(j_bbox_albedo[BBOX_MIN_KEY][0].get<float>(),
                                               j_bbox_albedo[BBOX_MIN_KEY][1].get<float>(),
                                               j_bbox_albedo[BBOX_MIN_KEY][2].get<float>());
    material->albedo_grid.bbox.max = glm::vec3(j_bbox_albedo[BBOX_MAX_KEY][0].get<float>(),
                                               j_bbox_albedo[BBOX_MAX_KEY][1].get<float>(),
                                               j_bbox_albedo[BBOX_MAX_KEY][2].get<float>());

    // --- Emission grid ---
    const auto& j_emission         = medium_node[EMISSION_GRID_KEY];
    material->emission_grid.handle = (AssetHandle)j_emission[GRID_KEY].get<uint64_t>();
    material->emission_grid.scale  = j_emission[SCALE_KEY].get<float>();

    const auto& j_bbox_emission      = j_emission[BBOX_KEY];
    material->emission_grid.bbox.min = glm::vec3(j_bbox_emission[BBOX_MIN_KEY][0].get<float>(),
                                                 j_bbox_emission[BBOX_MIN_KEY][1].get<float>(),
                                                 j_bbox_emission[BBOX_MIN_KEY][2].get<float>());
    material->emission_grid.bbox.max = glm::vec3(j_bbox_emission[BBOX_MAX_KEY][0].get<float>(),
                                                 j_bbox_emission[BBOX_MAX_KEY][1].get<float>(),
                                                 j_bbox_emission[BBOX_MAX_KEY][2].get<float>());

    // component.g = j[HETEROGENEOUS_MEDIUM_KEY][G_KEY].get<float>();

    return material;
}

bool MediumGUIRenderer<HeterogeneousMedium>::renderGUI(const atcg::ref_ptr<HeterogeneousMedium>& medium,
                                                       const std::string& key,
                                                       bool& deactivated)
{
#ifndef ATCG_HEADLESS
    bool updated = false;

    ImGui::Text("Density");
    auto new_handle = Utils::displayTexture3DSelection("densitytexture3d", medium->density_grid.handle, deactivated);
    updated         = updated || (new_handle != medium->density_grid.handle);
    medium->density_grid.handle = new_handle;
    updated = ImGui::DragFloat("Desity Scale##texture3d", &medium->density_grid.scale, 0.01f, 0.0f, 10.0f) || updated;
    ImGui::Text("Bounding Box");
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Min##density", glm::value_ptr(medium->density_grid.bbox.min), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Max##density", glm::value_ptr(medium->density_grid.bbox.max), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    ImGui::Separator();
    ImGui::Text("Albedo");
    new_handle = Utils::displayTexture3DSelection("albedotexture3d", medium->albedo_grid.handle, deactivated);
    updated    = updated || (new_handle != medium->albedo_grid.handle);
    medium->albedo_grid.handle = new_handle;
    updated     = ImGui::DragFloat("Albedo Scale##texture3d", &medium->albedo_grid.scale, 0.01f, 0.0f, 1.0f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    ImGui::Text("Bounding Box");
    updated     = ImGui::DragFloat3("Min##albedo", glm::value_ptr(medium->albedo_grid.bbox.min), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Max##albedo", glm::value_ptr(medium->albedo_grid.bbox.max), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    ImGui::Separator();
    ImGui::Text("Emission");
    new_handle = Utils::displayTexture3DSelection("emissiontexture3d", medium->emission_grid.handle, deactivated);
    updated    = updated || (new_handle != medium->emission_grid.handle);
    medium->emission_grid.handle = new_handle;
    updated =
        ImGui::DragFloat("Emission Scale##texture3d", &medium->emission_grid.scale, 0.01f, 0.0f, 10.0f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    ImGui::Text("Bounding Box");
    updated     = ImGui::DragFloat3("Min##emission", glm::value_ptr(medium->emission_grid.bbox.min), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Max##emission", glm::value_ptr(medium->emission_grid.bbox.max), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    // updated     = ImGui::DragFloat("g##het", &medium->g, 0.01f, -1.0f, 1.0f) || updated;
    // deactivated = ImGui::IsItemDeactivated() || deactivated;
#endif
    return updated;
}

void HeterogeneousMedium::registerMedium(MediumRegistry::Registry* registry)
{
    ATCG_REGISTER_MEDIUM(registry, "Heterogeneous", HeterogeneousMedium);
}

}    // namespace atcg