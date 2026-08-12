#pragma once

#include <Asset/Asset.h>
#include <Core/API.h>
#include <DataStructure/Dictionary.h>
#include <Material/MediumFlags.h>
#include <json.hpp>
#include <filesystem>

namespace atcg
{
class ATCG_API Medium : public Asset
{
public:
    Medium(const std::string& type, const Dictionary& dict);

    virtual ~Medium() {}

    ATCG_INLINE static AssetType getStaticType() { return AssetType::Medium; }

    ATCG_INLINE virtual AssetType getType() const override { return getStaticType(); }

    ATCG_INLINE const std::string& getMediumType() const { return _medium_type; }

    ATCG_INLINE const MediumFlags& getFlags() const { return _flags; }

    virtual atcg::ref_ptr<Medium> clone() const = 0;

protected:
    std::string _medium_type;
    MediumFlags _flags = MediumFlags::None;
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