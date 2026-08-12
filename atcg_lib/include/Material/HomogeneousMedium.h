#pragma once

#include <Material/Medium.h>
#include <Material/MediumRegistry.h>

namespace atcg
{
class HomogeneousMedium : public Medium
{
public:
    HomogeneousMedium(const Dictionary& dict);

    ATCG_INLINE void setAlbedo(const glm::vec3& albedo) { _albedo = albedo; }

    ATCG_INLINE void setDensity(const float density) { _density = density; }

    ATCG_INLINE void setLe(const float Le) { _Le = Le; }

    ATCG_INLINE void setLeColor(const glm::vec3& Le_color) { _Le_color = Le_color; }

    ATCG_INLINE glm::vec3 albedo() const { return _albedo; }

    ATCG_INLINE float density() const { return _density; }

    ATCG_INLINE float Le() const { return _Le; }

    ATCG_INLINE glm::vec3 Le_color() const { return _Le_color; }

    virtual atcg::ref_ptr<Medium> clone() const override;

    static void registerMedium(MediumRegistry::Registry* registry);

private:
    glm::vec3 _albedo   = glm::vec3(0);
    float _density      = 0.0f;
    float _Le           = 0.0f;
    glm::vec3 _Le_color = glm::vec3(1);
};

template<>
struct ATCG_API MediumSerializer<HomogeneousMedium>
{
    static void serialize(const atcg::ref_ptr<HomogeneousMedium>& medium, const std::filesystem::path& path);

    static atcg::ref_ptr<HomogeneousMedium> deserialize(const std::filesystem::path& path,
                                                        const nlohmann::json& medium_node);
};

template<>
struct ATCG_API MediumGUIRenderer<HomogeneousMedium>
{
    static bool renderGUI(const atcg::ref_ptr<HomogeneousMedium>& medium, const std::string& key, bool& deactivated);
};

}    // namespace atcg