#pragma once

#include <Core/API.h>
#include <Core/Assert.h>
#include <Renderer/Texture.h>
#include <Asset/Asset.h>
#include <DataStructure/Dictionary.h>
#include <DataStructure/Registry.h>

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
    Material(const std::string& type);

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

    virtual atcg::ref_ptr<Material> clone() const = 0;

    ATCG_INLINE static AssetType getStaticType() { return AssetType::Material; }

    ATCG_INLINE virtual AssetType getType() const override { return getStaticType(); }

    ATCG_INLINE const std::string& getMaterialType() const { return _material_type; };


protected:
    std::array<uint32_t, 5> _used_texture_ids;
    bool _uploaded = false;
    std::string _material_type;
};

using MaterialBuilder           = std::function<atcg::ref_ptr<Material>(const Dictionary&)>;
using MaterialGUIFunction       = std::function<bool(const atcg::ref_ptr<Material>&, const std::string&, bool&)>;
using MaterialSerializeFunction = std::function<void(const atcg::ref_ptr<Material>&, const std::filesystem::path&)>;
using MaterialDeserializeFunction =
    std::function<atcg::ref_ptr<Material>(const std::filesystem::path&, const nlohmann::json&)>;

namespace MaterialRegistry
{
struct MaterialFunctions
{
    MaterialBuilder builder;
    MaterialGUIFunction gui_function;
    MaterialSerializeFunction serializer_function;
    MaterialDeserializeFunction deserializer_function;
};

using Registry = atcg::Registry<MaterialFunctions>;

ATCG_API void registerMaterial(Registry* registry, std::string_view type, MaterialFunctions);

ATCG_API atcg::ref_ptr<Material> createMaterial(Registry* registry, const std::string& type, const Dictionary& dict);

ATCG_API bool renderMaterialGUI(Registry* registry,
                                const atcg::ref_ptr<Material>& material,
                                const std::string& key,
                                bool& deactivated);

ATCG_API const std::vector<std::string>& getRegisteredMaterialTypes(Registry* registry);

ATCG_API void
serializeMaterial(Registry* registry, const atcg::ref_ptr<Material>& material, const std::filesystem::path& path);

ATCG_API atcg::ref_ptr<Material> deserializeMaterial(Registry* registry,
                                                     std::string_view material_type,
                                                     const std::filesystem::path& path,
                                                     const nlohmann::json& material_node);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<MaterialRegistry::Registry>();
    ATCG_ASSERT(registry, "Material registry not found");
    return registry;
}

ATCG_INLINE void registerMaterial(std::string_view type, MaterialFunctions functions)
{
    registerMaterial(getRegistry(), type, std::move(functions));
}

ATCG_INLINE atcg::ref_ptr<Material> createMaterial(const std::string& type, const Dictionary& dict)
{
    return createMaterial(getRegistry(), type, dict);
}

ATCG_INLINE bool renderMaterialGUI(const atcg::ref_ptr<Material>& material, const std::string& key, bool& deactivated)
{
    return renderMaterialGUI(getRegistry(), material, key, deactivated);
}

ATCG_INLINE const std::vector<std::string>& getRegisteredMaterialTypes()
{
    return getRegisteredMaterialTypes(getRegistry());
}

ATCG_INLINE void serializeMaterial(const atcg::ref_ptr<Material>& material, const std::filesystem::path& path)
{
    serializeMaterial(getRegistry(), material, path);
}

ATCG_INLINE atcg::ref_ptr<Material> deserializeMaterial(std::string_view material_type,
                                                        const std::filesystem::path& path,
                                                        const nlohmann::json& material_node)
{
    return deserializeMaterial(getRegistry(), material_type, path, material_node);
}

}    // namespace MaterialRegistry

class ATCG_API MicrofacetMaterial : public Material
{
public:
    MicrofacetMaterial(const std::string& type);

    /**
     * @brief Get the diffuse texture.
     *
     * @return The diffuse texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getDiffuseTexture() const { return _diffuse_texture; }


    /**
     * @brief Get the roughness texture.
     *
     * @return The roughness texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getRoughnessTexture() const { return _roughness_texture; }

    /**
     * @brief Get the ior texture.
     *
     * @return The ior texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getIorTexture() const { return _ior_texture; }

    /**
     * @brief Set the diffuse texture.
     *
     * @param texture The diffuse texture
     */
    ATCG_INLINE void setDiffuseTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _diffuse_texture = texture; }

    /**
     * @brief Set the roughness texture.
     *
     * @param texture The roughness texture
     */
    ATCG_INLINE void setRoughnessTexture(const atcg::ref_ptr<atcg::Texture2D>& texture)
    {
        _roughness_texture = texture;
    }

    /**
     * @brief Set the ior texture.
     *
     * @param texture The ior texture
     */
    ATCG_INLINE void setIorTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _ior_texture = texture; }

    /**
     * @brief Set the diffuse color.
     *
     * @param color The color
     */
    void setDiffuseColor(const glm::vec4& color);

    /**
     * @brief Set the diffuse color.
     *
     * @param color The color
     */
    void setDiffuseColor(const glm::vec3& color);

    /**
     * @brief Set the roughness value.
     *
     * @param roughness The roughness
     */
    void setRoughness(const float roughness);

    /**
     * @brief Set the ior value.
     *
     * @param ior The ior value
     */
    void setIor(const float ior);

protected:
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture;
    atcg::ref_ptr<atcg::Texture2D> _roughness_texture;
    atcg::ref_ptr<atcg::Texture2D> _ior_texture;
};

class ATCG_API OpaqueMaterial : public MicrofacetMaterial
{
public:
    OpaqueMaterial();

    /**
     * @brief Get the normal texture.
     *
     * @return The normal texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getNormalTexture() const { return _normal_texture; }

    /**
     * @brief Get the metallic texture.
     *
     * @return The metallic texture
     */
    ATCG_INLINE atcg::ref_ptr<atcg::Texture2D> getMetallicTexture() const { return _metallic_texture; }

    /**
     * @brief Set the normal texture.
     *
     * @param texture The normal texture
     */
    ATCG_INLINE void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _normal_texture = texture; }

    /**
     * @brief Set the metallic texture.
     *
     * @param texture The metallic texture
     */
    ATCG_INLINE void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _metallic_texture = texture; }

    /**
     * @brief The the metallic value.
     *
     * @param metallic The metallic value
     */
    void setMetallic(const float metallic);

    /**
     * @brief Remove the normal map
     */
    void removeNormalMap();

    /**
     * @brief Upload the material to a shader
     *
     * @param renderer The renderer
     * @param shader The shader
     */
    virtual void uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader) override;

    virtual atcg::ref_ptr<Material> clone() const override;

    static void registerMaterial(MaterialRegistry::Registry* registry);

private:
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;
};

class ATCG_API DielectricMaterial : public MicrofacetMaterial
{
public:
    DielectricMaterial();

    /**
     * @brief Upload the material to a shader
     *
     * @param renderer The renderer
     * @param shader The shader
     */
    virtual void uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader) override;

    virtual atcg::ref_ptr<Material> clone() const override;

    static void registerMaterial(MaterialRegistry::Registry* registry);

private:
};

class ATCG_API NullMaterial : public Material
{
public:
    NullMaterial();

    /**
     * @brief Upload the material to a shader
     *
     * @param renderer The renderer
     * @param shader The shader
     */
    virtual void uploadMaterial(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader) override;

    virtual atcg::ref_ptr<Material> clone() const override;

    static void registerMaterial(MaterialRegistry::Registry* registry);

private:
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

template<>
struct ATCG_API MaterialSerializer<OpaqueMaterial>
{
    static void serialize(const atcg::ref_ptr<OpaqueMaterial>& material, const std::filesystem::path& path);

    static atcg::ref_ptr<OpaqueMaterial> deserialize(const std::filesystem::path& path,
                                                     const nlohmann::json& material_node);
};

template<>
struct ATCG_API MaterialSerializer<DielectricMaterial>
{
    static void serialize(const atcg::ref_ptr<DielectricMaterial>& material, const std::filesystem::path& path);

    static atcg::ref_ptr<DielectricMaterial> deserialize(const std::filesystem::path& path,
                                                         const nlohmann::json& material_node);
};

template<>
struct ATCG_API MaterialSerializer<NullMaterial>
{
    static void serialize(const atcg::ref_ptr<NullMaterial>& material, const std::filesystem::path& path);

    static atcg::ref_ptr<NullMaterial> deserialize(const std::filesystem::path& path,
                                                   const nlohmann::json& material_node);
};

template<typename T>
struct ATCG_API MaterialGUIRenderer
{
    static bool renderGUI(const atcg::ref_ptr<T>& material, const std::string& key, bool& deactivated) { return false; }
};

template<>
struct ATCG_API MaterialGUIRenderer<OpaqueMaterial>
{
    static bool renderGUI(const atcg::ref_ptr<OpaqueMaterial>& material, const std::string& key, bool& deactivated);
};

template<>
struct ATCG_API MaterialGUIRenderer<DielectricMaterial>
{
    static bool renderGUI(const atcg::ref_ptr<DielectricMaterial>& material, const std::string& key, bool& deactivated);
};

template<>
struct ATCG_API MaterialGUIRenderer<NullMaterial>
{
    static bool renderGUI(const atcg::ref_ptr<NullMaterial>& material, const std::string& key, bool& deactivated);
};


#define ATCG_REGISTER_MATERIAL(registry, MaterialType, MaterialClass)                                                  \
    {                                                                                                                  \
        atcg::MaterialRegistry::MaterialFunctions functions = {                                                        \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<MaterialClass>(); },                              \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::string& key, bool& deactivated)               \
            {                                                                                                          \
                return atcg::MaterialGUIRenderer<MaterialClass>::renderGUI(                                            \
                    std::dynamic_pointer_cast<MaterialClass>(material),                                                \
                    key,                                                                                               \
                    deactivated);                                                                                      \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::filesystem::path& path)                       \
            {                                                                                                          \
                atcg::MaterialSerializer<MaterialClass>::serialize(std::dynamic_pointer_cast<MaterialClass>(material), \
                                                                   path);                                              \
            },                                                                                                         \
            [](const std::filesystem::path& path, const nlohmann::json& material_node)                                 \
            { return atcg::MaterialSerializer<MaterialClass>::deserialize(path, material_node); }};                    \
        registry->registerType(MaterialType, std::move(functions));                                                    \
    }

#define ATCG_REGISTER_MATERIAL_PLUGIN(registry, handle, MaterialType, MaterialClass)                                   \
    {                                                                                                                  \
        atcg::MaterialRegistry::MaterialFunctions functions = {                                                        \
            [](const atcg::Dictionary& dict) { return atcg::make_ref<MaterialClass>(); },                              \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::string& key, bool& deactivated)               \
            {                                                                                                          \
                return atcg::MaterialGUIRenderer<MaterialClass>::renderGUI(                                            \
                    std::dynamic_pointer_cast<MaterialClass>(material),                                                \
                    key,                                                                                               \
                    deactivated);                                                                                      \
            },                                                                                                         \
            [](const atcg::ref_ptr<atcg::Material>& material, const std::filesystem::path& path)                       \
            {                                                                                                          \
                atcg::MaterialSerializer<MaterialClass>::serialize(std::dynamic_pointer_cast<MaterialClass>(material), \
                                                                   path);                                              \
            },                                                                                                         \
            [](const std::filesystem::path& path, const nlohmann::json& material_node)                                 \
            { return atcg::MaterialSerializer<MaterialClass>::deserialize(path, material_node); }};                    \
        registry->registerType(handle, MaterialType, std::move(functions));                                            \
    }

}    // namespace atcg