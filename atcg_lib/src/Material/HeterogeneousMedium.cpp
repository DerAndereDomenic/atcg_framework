#include <Material/HeterogeneousMedium.h>

#include <Material/HeterogeneousMediumData.h>

#include <Asset/AssetManagerSystem.h>
#include <Core/glm.h>
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
class HeterogeneousMedium::Impl
{
public:
    HeterogeneousMedium::Impl(const atcg::Dictionary& dict);

    ~Impl();

    // CPU interface
    atcg::ref_ptr<Texture3D> _default_emission_texture;
    atcg::ref_ptr<Texture3D> _default_albedo_texture;

    GridComponent density_grid;
    GridComponent albedo_grid;
    GridComponent emission_grid;

    // GPU interface
    atcg::ref_ptr<Texture3D> _density_texture_gpu;
    atcg::ref_ptr<Texture3D> _albedo_texture_gpu;
    atcg::ref_ptr<Texture3D> _emission_texture_gpu;

    atcg::dref_ptr<HeterogeneousMediumData> _medium_data_buffer;
};

HeterogeneousMedium::Impl::Impl(const atcg::Dictionary& dict)
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

HeterogeneousMedium::Impl::~Impl() {}

HeterogeneousMedium::HeterogeneousMedium(const Dictionary& dict) : Medium("Heterogeneous", dict)
{
    impl = std::make_unique<Impl>(dict);

    _flags = MediumFlag::Heterogeneous;
}

HeterogeneousMedium::~HeterogeneousMedium() {}

atcg::ref_ptr<Texture3D> HeterogeneousMedium::density() const
{
    return AssetManager::getAsset<Texture3D>(densityGrid().handle);
}

atcg::ref_ptr<Texture3D> HeterogeneousMedium::albedo() const
{
    return AssetManager::isAssetHandleValid(albedoGrid().handle)
               ? AssetManager::getAsset<Texture3D>(albedoGrid().handle)
               : impl->_default_albedo_texture;
}

atcg::ref_ptr<Texture3D> HeterogeneousMedium::emission() const
{
    return AssetManager::isAssetHandleValid(emissionGrid().handle)
               ? AssetManager::getAsset<Texture3D>(emissionGrid().handle)
               : impl->_default_emission_texture;
}

void HeterogeneousMedium::uploadMedium(RendererSystem* renderer,
                                       const atcg::ref_ptr<Shader>& shader,
                                       const glm::mat4& model)
{
    uint32_t albedo_id   = renderer->popTextureID();
    _texture_ids[0]      = albedo_id;
    uint32_t density_id  = renderer->popTextureID();
    _texture_ids[1]      = density_id;
    uint32_t emission_id = renderer->popTextureID();
    _texture_ids[2]      = emission_id;

    if(density())
    {
        shader->setInt("density_grid", density_id);
        shader->setFloat("density_scale", densityGrid().scale);
        glm::mat4 to_uvw = glm::mat4(1);
        glm::vec3 scale  = densityGrid().bbox.max - densityGrid().bbox.min;
        to_uvw           = to_uvw * glm::scale(1.0f / scale);
        to_uvw           = to_uvw * glm::translate(-densityGrid().bbox.min);
        to_uvw           = to_uvw * glm::inverse(model);
        shader->setMat4("density_to_uvw", to_uvw);
        GraphicsCommand::bindTexture(density_id, density());
    }

    if(albedo())
    {
        shader->setInt("albedo_grid", albedo_id);
        shader->setFloat("albedo_scale", albedoGrid().scale);
        glm::mat4 to_uvw = glm::mat4(1);
        glm::vec3 scale  = albedoGrid().bbox.max - albedoGrid().bbox.min;
        to_uvw           = to_uvw * glm::scale(1.0f / scale);
        to_uvw           = to_uvw * glm::translate(-albedoGrid().bbox.min);
        to_uvw           = to_uvw * glm::inverse(model);
        shader->setMat4("albedo_to_uvw", to_uvw);
        GraphicsCommand::bindTexture(albedo_id, albedo());
    }

    if(emission())
    {
        shader->setInt("emission_grid", emission_id);
        shader->setFloat("emission_scale", emissionGrid().scale);
        glm::mat4 to_uvw = glm::mat4(1);
        glm::vec3 scale  = emissionGrid().bbox.max - emissionGrid().bbox.min;
        to_uvw           = to_uvw * glm::scale(1.0f / scale);
        to_uvw           = to_uvw * glm::translate(-emissionGrid().bbox.min);
        to_uvw           = to_uvw * glm::inverse(model);
        shader->setMat4("emission_to_uvw", to_uvw);
        GraphicsCommand::bindTexture(emission_id, emission());
    }

    _uploaded = true;
}

void HeterogeneousMedium::updateData()
{
    HeterogeneousMediumData data;

    // TODO: world_to_local

    auto density_grid          = densityGrid();
    auto density_texture       = AssetManager::getAsset<Texture3D>(density_grid.handle)->clone();
    impl->_density_texture_gpu = std::static_pointer_cast<Texture3D>(density_texture);
    auto emission_grid         = emissionGrid();
    auto emission_texture      = AssetManager::getAsset<Texture3D>(emission_grid.handle);
    impl->_emission_texture_gpu =
        emission_texture ? std::static_pointer_cast<Texture3D>(emission_texture->clone()) : nullptr;
    auto albedo_grid          = albedoGrid();
    auto albedo_texture       = AssetManager::getAsset<Texture3D>(albedo_grid.handle);
    impl->_albedo_texture_gpu = albedo_texture ? std::static_pointer_cast<Texture3D>(albedo_texture->clone()) : nullptr;

    auto density_majorant = impl->_density_texture_gpu->getData(atcg::GPU).max().item<float>();

    data.density_grid.storage.texture = impl->_density_texture_gpu->getTextureObject();
    data.density_grid.scale           = density_grid.scale;
    data.density_majorant             = density_majorant * data.density_grid.scale;
    {
        glm::mat4 to_uvw = glm::mat4(1);
        glm::vec3 scale  = density_grid.bbox.max - density_grid.bbox.min;
        to_uvw           = to_uvw * glm::scale(1.0f / scale);
        to_uvw           = to_uvw * glm::translate(-density_grid.bbox.min);
        // to_uvw                   = to_uvw * world_to_local;
        data.density_grid.to_uvw = to_uvw;
    }

    data.emission_grid.storage.texture =
        impl->_emission_texture_gpu ? impl->_emission_texture_gpu->getTextureObject() : 0;
    data.emission_grid.default_value = glm::vec3(0);
    data.emission_grid.scale         = emission_grid.scale;
    {
        glm::mat4 to_uvw = glm::mat4(1);
        glm::vec3 scale  = emission_grid.bbox.max - emission_grid.bbox.min;
        to_uvw           = to_uvw * glm::scale(1.0f / scale);
        to_uvw           = to_uvw * glm::translate(-emission_grid.bbox.min);
        // to_uvw                    = to_uvw * world_to_local;
        data.emission_grid.to_uvw = to_uvw;
    }

    data.albedo_grid.storage.texture = impl->_albedo_texture_gpu ? impl->_albedo_texture_gpu->getTextureObject() : 0;
    data.albedo_grid.scale           = albedo_grid.scale;
    {
        glm::mat4 to_uvw = glm::mat4(1);
        glm::vec3 scale  = albedo_grid.bbox.max - albedo_grid.bbox.min;
        to_uvw           = to_uvw * glm::scale(1.0f / scale);
        to_uvw           = to_uvw * glm::translate(-albedo_grid.bbox.min);
        // to_uvw                  = to_uvw * world_to_local;
        data.albedo_grid.to_uvw = to_uvw;
    }

    impl->_medium_data_buffer.upload(&data);
}

void HeterogeneousMedium::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                             const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    updateData();
    if(_phase_function != nullptr) _phase_function->ensureInitialized(pipeline, sbt);

    const std::string ptx_filename = "./bin/HeterogeneousMedium_ptx.ptx";
    OptixProgramGroup eval_transmittance_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_evalTransmittance"});
    OptixProgramGroup sample_medium_event_prog_group =
        pipeline->addCallableShader({ptx_filename, "__direct_callable__heterogeneousMedium_sampleMediumEvent"});

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

HeterogeneousMedium::GridComponent& HeterogeneousMedium::densityGrid() const
{
    return impl->density_grid;
}

HeterogeneousMedium::GridComponent& HeterogeneousMedium::albedoGrid() const
{
    return impl->albedo_grid;
}

HeterogeneousMedium::GridComponent& HeterogeneousMedium::emissionGrid() const
{
    return impl->emission_grid;
}

atcg::ref_ptr<Medium> HeterogeneousMedium::clone() const
{
    auto medium = atcg::make_ref<HeterogeneousMedium>(atcg::Dictionary());

    medium->densityGrid()  = densityGrid();
    medium->albedoGrid()   = albedoGrid();
    medium->emissionGrid() = emissionGrid();

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

    j_density_grid[GRID_KEY]  = (uint64_t)medium->densityGrid().handle;
    j_density_grid[SCALE_KEY] = medium->densityGrid().scale;
    nlohmann::json j_bbox_density;
    j_bbox_density[BBOX_MIN_KEY] = nlohmann::json::array(
        {medium->densityGrid().bbox.min.x, medium->densityGrid().bbox.min.y, medium->densityGrid().bbox.min.z});
    j_bbox_density[BBOX_MAX_KEY] = nlohmann::json::array(
        {medium->densityGrid().bbox.max.x, medium->densityGrid().bbox.max.y, medium->densityGrid().bbox.max.z});
    j_density_grid[BBOX_KEY] = j_bbox_density;

    j_albedo_grid[GRID_KEY]  = (uint64_t)medium->albedoGrid().handle;
    j_albedo_grid[SCALE_KEY] = medium->albedoGrid().scale;
    nlohmann::json j_bbox_albedo;
    j_bbox_albedo[BBOX_MIN_KEY] = nlohmann::json::array(
        {medium->albedoGrid().bbox.min.x, medium->albedoGrid().bbox.min.y, medium->albedoGrid().bbox.min.z});
    j_bbox_albedo[BBOX_MAX_KEY] = nlohmann::json::array(
        {medium->albedoGrid().bbox.max.x, medium->albedoGrid().bbox.max.y, medium->albedoGrid().bbox.max.z});
    j_albedo_grid[BBOX_KEY] = j_bbox_albedo;

    j_emission_grid[GRID_KEY]  = (uint64_t)medium->emissionGrid().handle;
    j_emission_grid[SCALE_KEY] = medium->emissionGrid().scale;
    nlohmann::json j_bbox_emission;
    j_bbox_emission[BBOX_MIN_KEY] = nlohmann::json::array(
        {medium->emissionGrid().bbox.min.x, medium->emissionGrid().bbox.min.y, medium->emissionGrid().bbox.min.z});
    j_bbox_emission[BBOX_MAX_KEY] = nlohmann::json::array(
        {medium->emissionGrid().bbox.max.x, medium->emissionGrid().bbox.max.y, medium->emissionGrid().bbox.max.z});
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
    const auto& j_density          = medium_node[DENSITY_GRID_KEY];
    material->densityGrid().handle = (AssetHandle)j_density[GRID_KEY].get<uint64_t>();
    material->densityGrid().scale  = j_density[SCALE_KEY].get<float>();

    const auto& j_bbox_density       = j_density[BBOX_KEY];
    material->densityGrid().bbox.min = glm::vec3(j_bbox_density[BBOX_MIN_KEY][0].get<float>(),
                                                 j_bbox_density[BBOX_MIN_KEY][1].get<float>(),
                                                 j_bbox_density[BBOX_MIN_KEY][2].get<float>());
    material->densityGrid().bbox.max = glm::vec3(j_bbox_density[BBOX_MAX_KEY][0].get<float>(),
                                                 j_bbox_density[BBOX_MAX_KEY][1].get<float>(),
                                                 j_bbox_density[BBOX_MAX_KEY][2].get<float>());

    // --- Albedo grid ---
    const auto& j_albedo          = medium_node[ALBEDO_GRID_KEY];
    material->albedoGrid().handle = (AssetHandle)j_albedo[GRID_KEY].get<uint64_t>();
    material->albedoGrid().scale  = j_albedo[SCALE_KEY].get<float>();

    const auto& j_bbox_albedo       = j_albedo[BBOX_KEY];
    material->albedoGrid().bbox.min = glm::vec3(j_bbox_albedo[BBOX_MIN_KEY][0].get<float>(),
                                                j_bbox_albedo[BBOX_MIN_KEY][1].get<float>(),
                                                j_bbox_albedo[BBOX_MIN_KEY][2].get<float>());
    material->albedoGrid().bbox.max = glm::vec3(j_bbox_albedo[BBOX_MAX_KEY][0].get<float>(),
                                                j_bbox_albedo[BBOX_MAX_KEY][1].get<float>(),
                                                j_bbox_albedo[BBOX_MAX_KEY][2].get<float>());

    // --- Emission grid ---
    const auto& j_emission          = medium_node[EMISSION_GRID_KEY];
    material->emissionGrid().handle = (AssetHandle)j_emission[GRID_KEY].get<uint64_t>();
    material->emissionGrid().scale  = j_emission[SCALE_KEY].get<float>();

    const auto& j_bbox_emission       = j_emission[BBOX_KEY];
    material->emissionGrid().bbox.min = glm::vec3(j_bbox_emission[BBOX_MIN_KEY][0].get<float>(),
                                                  j_bbox_emission[BBOX_MIN_KEY][1].get<float>(),
                                                  j_bbox_emission[BBOX_MIN_KEY][2].get<float>());
    material->emissionGrid().bbox.max = glm::vec3(j_bbox_emission[BBOX_MAX_KEY][0].get<float>(),
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
    auto new_handle = Utils::displayTexture3DSelection("densitytexture3d", medium->densityGrid().handle, deactivated);
    updated         = updated || (new_handle != medium->densityGrid().handle);
    medium->densityGrid().handle = new_handle;
    updated = ImGui::DragFloat("Desity Scale##texture3d", &medium->densityGrid().scale, 0.01f, 0.0f, 10.0f) || updated;
    ImGui::Text("Bounding Box");
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Min##density", glm::value_ptr(medium->densityGrid().bbox.min), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Max##density", glm::value_ptr(medium->densityGrid().bbox.max), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    ImGui::Separator();
    ImGui::Text("Albedo");
    new_handle = Utils::displayTexture3DSelection("albedotexture3d", medium->albedoGrid().handle, deactivated);
    updated    = updated || (new_handle != medium->albedoGrid().handle);
    medium->albedoGrid().handle = new_handle;
    updated = ImGui::DragFloat("Albedo Scale##texture3d", &medium->albedoGrid().scale, 0.01f, 0.0f, 1.0f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    ImGui::Text("Bounding Box");
    updated     = ImGui::DragFloat3("Min##albedo", glm::value_ptr(medium->albedoGrid().bbox.min), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Max##albedo", glm::value_ptr(medium->albedoGrid().bbox.max), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;

    ImGui::Separator();
    ImGui::Text("Emission");
    new_handle = Utils::displayTexture3DSelection("emissiontexture3d", medium->emissionGrid().handle, deactivated);
    updated    = updated || (new_handle != medium->emissionGrid().handle);
    medium->emissionGrid().handle = new_handle;
    updated =
        ImGui::DragFloat("Emission Scale##texture3d", &medium->emissionGrid().scale, 0.01f, 0.0f, 10.0f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    ImGui::Text("Bounding Box");
    updated     = ImGui::DragFloat3("Min##emission", glm::value_ptr(medium->emissionGrid().bbox.min), 0.05f) || updated;
    deactivated = ImGui::IsItemDeactivated() || deactivated;
    updated     = ImGui::DragFloat3("Max##emission", glm::value_ptr(medium->emissionGrid().bbox.max), 0.05f) || updated;
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