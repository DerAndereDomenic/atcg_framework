#pragma once

#include <Asset/Asset.h>
#include <Core/RaytracingComponent.h>
#include <Core/API.h>
#include <DataStructure/Dictionary.h>
#include <Material/MediumFlags.h>
#include <Material/MediumVPtrTable.h>
#include <Renderer/Renderer.h>

#include <json.hpp>
#include <filesystem>
#include <array>

namespace atcg
{
class ATCG_API Medium : public Asset, public RaytracingComponent
{
public:
    Medium(const std::string& type, const Dictionary& dict);

    virtual ~Medium() {}

    ATCG_INLINE static AssetType getStaticType() { return AssetType::Medium; }

    ATCG_INLINE virtual AssetType getType() const override { return getStaticType(); }

    ATCG_INLINE const std::string& getMediumType() const { return _medium_type; }

    ATCG_INLINE const MediumFlag& flags() const { return _flags; }

    virtual atcg::ref_ptr<Medium> clone() const = 0;

    virtual void
    uploadMedium(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader, const glm::mat4& model) = 0;

    void releaseTextureIDs(RendererSystem* renderer);

    ATCG_INLINE const MediumVPtrTable* getVPtrTable() const { return _medium_vptr_table.get(); }

protected:
    std::array<uint32_t, 3> _texture_ids;
    bool _uploaded = false;
    std::string _medium_type;
    MediumFlag _flags = MediumFlag::None;

    atcg::dref_ptr<MediumVPtrTable> _medium_vptr_table;
};

template<typename T>
struct ATCG_API MediumSerializer
{
    static void serialize(const atcg::ref_ptr<T>& medium, const std::filesystem::path& path) {}

    static atcg::ref_ptr<T> deserialize(const std::filesystem::path& path, const nlohmann::json& medium_node)
    {
        return nullptr;
    }
};

template<typename T>
struct ATCG_API MediumGUIRenderer
{
    static bool renderGUI(const atcg::ref_ptr<T>& medium, const std::string& key, bool& deactivated) { return false; }
};

}    // namespace atcg