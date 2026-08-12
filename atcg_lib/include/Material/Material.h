#pragma once

#include <Core/API.h>
#include <Core/Assert.h>
#include <Renderer/Texture.h>
#include <Asset/Asset.h>
#include <DataStructure/Dictionary.h>
#include <Material/MaterialFlags.h>

#include <json.hpp>
#include <filesystem>

namespace atcg
{

class RendererSystem;
class Shader;

/**
 * @brief A class to model a material.
 */
struct ATCG_API Material : public Asset
{
    /**
     * @brief Constructor
     */
    Material(const std::string& type, const atcg::Dictionary& dict);

    virtual ~Material() {}

    /**
     * @brief Upload the material to a shader
     *
     * @param renderer The renderer
     * @param shader The shader
     */
    virtual void uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader) = 0;

    /**
     * @brief Release used texture units after an upload.
     * Should only be called after uploadMaterial was called
     *
     * @param renderer The renderer
     */
    void releaseTextureIDs(RendererSystem* renderer);

    /**
     * @brief Create a deep copy of the material.
     *
     * @return The cloned material.
     */
    virtual atcg::ref_ptr<Material> clone() const = 0;

    /**
     * @brief Get the type of the asset
     *
     * @return The asset type
     */
    ATCG_INLINE static AssetType getStaticType() { return AssetType::Material; }

    /**
     * @brief Get the type of the asset
     *
     * @return The asset type
     */
    ATCG_INLINE virtual AssetType getType() const override { return getStaticType(); }

    /**
     * @brief Get the material type
     *
     * @return The material type
     */
    ATCG_INLINE const std::string& getMaterialType() const { return _material_type; };

    /**
     * @brief Get the material flags
     *
     * @return The material flags
     */
    ATCG_INLINE const MaterialFlag& getMaterialFlags() const { return _flags; }

protected:
    std::array<uint32_t, 5> _used_texture_ids;
    bool _uploaded = false;
    std::string _material_type;
    MaterialFlag _flags = MaterialFlag::None;
};

template<typename T>
struct ATCG_API MaterialSerializer
{
    static void serialize(const atcg::ref_ptr<T>& material, const std::filesystem::path& path) {}

    static atcg::ref_ptr<T> deserialize(const std::filesystem::path& path, const nlohmann::json& material_node)
    {
        return nullptr;
    }
};

template<typename T>
struct ATCG_API MaterialGUIRenderer
{
    static bool renderGUI(const atcg::ref_ptr<T>& material, const std::string& key, bool& deactivated) { return false; }
};

}    // namespace atcg