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
class ShaderManagerSystem
{
public:
    ShaderManagerSystem() = default;

    /**
     * @brief Add a shader
     *
     * @param name Name of the shader
     * @param shader The shader
     */
    void addShader(const std::string& name, const atcg::ref_ptr<Shader>& shader);

    /**
     * @brief Add a shader by loading it from file
     *
     * @param name The name of the .vs, .fs and optionally .gs file (without file ending)
     */
    void addShaderFromName(const std::string& name);

    /**
     * @brief Add a compute shader by loading it from file
     *
     * @param name The name of the .glsl file (without file ending)
     *
     */
    void addComputeShaderFromName(const std::string& name);
    /**
     * @brief Get the Shader object
     *
     * @param name The name
     * @return const atcg::ref_ptr<Shader>& The shader
     */
    const atcg::ref_ptr<Shader>& getShader(const std::string& name);

    /**
     * @brief Check if the shader exists
     *
     * @param name The name
     * @return True if the shader exists in the shader manager
     */
    bool hasShader(const std::string& name);

    /**
     * @brief This gets called by the application. Don't call manually
     */
    void onUpdate();

private:
    std::unordered_map<std::string, atcg::ref_ptr<Shader>> _shader;
    std::unordered_map<std::string, std::filesystem::file_time_type> _time_stamps;
};

/**
 * @brief This namespace encapuslates the default shader manager
 */
namespace ShaderManager
{
/**
 * @brief Add a shader
 *
 * @param name Name of the shader
 * @param shader The shader
 */
ATCG_INLINE void addShader(const std::string& name, const atcg::ref_ptr<Shader>& shader)
{
    SystemRegistry::instance()->getSystem<ShaderManagerSystem>()->addShader(name, shader);
}

/**
 * @brief Add a shader by loading it from file
 *
 * @param name The name of the .vs, .fs and optionally .gs file (without file ending)
 */
ATCG_INLINE void addShaderFromName(const std::string& name)
{
    SystemRegistry::instance()->getSystem<ShaderManagerSystem>()->addShaderFromName(name);
}

/**
 * @brief Add a compute shader by loading it from file
 *
 * @param name The name of the .glsl file (without file ending)
 *
 */
ATCG_INLINE void addComputeShaderFromName(const std::string& name)
{
    SystemRegistry::instance()->getSystem<ShaderManagerSystem>()->addComputeShaderFromName(name);
}

/**
 * @brief Get the Shader object
 *
 * @param name The name
 * @return const atcg::ref_ptr<Shader>& The shader
 */
ATCG_INLINE const atcg::ref_ptr<Shader>& getShader(const std::string& name)
{
    return SystemRegistry::instance()->getSystem<ShaderManagerSystem>()->getShader(name);
}

/**
 * @brief Check if the shader exists
 *
 * @param name The name
 * @return True if the shader exists in the shader manager
 */
ATCG_INLINE bool hasShader(const std::string& name)
{
    return SystemRegistry::instance()->getSystem<ShaderManagerSystem>()->hasShader(name);
}

/**
 * @brief This gets called by the application. Don't call manually
 */
ATCG_INLINE void onUpdate()
{
    SystemRegistry::instance()->getSystem<ShaderManagerSystem>()->onUpdate();
}
}    // namespace ShaderManager
}    // namespace atcg