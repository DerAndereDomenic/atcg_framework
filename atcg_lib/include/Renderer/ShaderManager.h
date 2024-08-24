#pragma once

#include <Core/Memory.h>
#include <Core/SystemRegistry.h>

#include <string>
#include <unordered_map>
#include <filesystem>

namespace atcg
{
class Shader;

/**
 * @brief This class manages all shaders used for the application
 * When using custom shaders, you have to add them via 'addShader' to get hot reloading or
 * use addShaderFromPath to also handle direct shader loading
 */
class ShaderManager
{
public:
    ShaderManager() = default;

    /**
     * @brief Add a shader
     *
     * @param name Name of the shader
     * @param shader The shader
     */
    ATCG_INLINE static void addShader(const std::string& name, const atcg::ref_ptr<Shader>& shader)
    {
        SystemRegistry::instance()->getSystem<ShaderManager>()->addShaderImpl(name, shader);
    }

    /**
     * @brief Add a shader by loading it from file
     *
     * @param name The name of the .vs, .fs and optionally .gs file (without file ending)
     */
    ATCG_INLINE static void addShaderFromName(const std::string& name)
    {
        SystemRegistry::instance()->getSystem<ShaderManager>()->addShaderFromNameImpl(name);
    }

    /**
     * @brief Add a compute shader by loading it from file
     *
     * @param name The name of the .glsl file (without file ending)
     *
     */
    ATCG_INLINE static void addComputerShaderFromName(const std::string& name)
    {
        SystemRegistry::instance()->getSystem<ShaderManager>()->addComputeShaderFromNameImpl(name);
    }

    /**
     * @brief Get the Shader object
     *
     * @param name The name
     * @return const atcg::ref_ptr<Shader>& The shader
     */
    ATCG_INLINE static const atcg::ref_ptr<Shader>& getShader(const std::string& name)
    {
        return SystemRegistry::instance()->getSystem<ShaderManager>()->getShaderImpl(name);
    }

    /**
     * @brief Check if the shader exists
     *
     * @param name The name
     * @return True if the shader exists in the shader manager
     */
    ATCG_INLINE static bool hasShader(const std::string& name)
    {
        return SystemRegistry::instance()->getSystem<ShaderManager>()->hasShaderImpl(name);
    }

    /**
     * @brief This gets called by the application. Don't call manually
     */
    ATCG_INLINE static void onUpdate() { SystemRegistry::instance()->getSystem<ShaderManager>()->onUpdateImpl(); }

private:
    void addShaderImpl(const std::string& name, const atcg::ref_ptr<Shader>& shader);
    void addShaderFromNameImpl(const std::string& name);
    void addComputeShaderFromNameImpl(const std::string& name);
    bool hasShaderImpl(const std::string& name);
    const atcg::ref_ptr<Shader>& getShaderImpl(const std::string& name);
    void onUpdateImpl();

    std::unordered_map<std::string, atcg::ref_ptr<Shader>> _shader;
    std::unordered_map<std::string, std::filesystem::file_time_type> _time_stamps;
};
}    // namespace atcg