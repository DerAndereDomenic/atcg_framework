#pragma once

#include <Core/Memory.h>

#include <string>
#include <unordered_map>

namespace atcg
{
class RaytracingShader;

/**
 * @brief This class manages all ray tracing shaders used for the application
 * When using custom shaders, you have to add them via 'addShader'.
 */
class RaytracingShaderManager
{
public:
    /**
     * @brief Add a shader
     *
     * @param name Name of the shader
     * @param shader The shader
     */
    static void addShader(const std::string& name, const atcg::ref_ptr<RaytracingShader>& shader);

    /**
     * @brief Get the Shader object
     *
     * @param name The name
     * @return const atcg::ref_ptr<Shader>& The shader
     */
    static const atcg::ref_ptr<RaytracingShader>& getShader(const std::string& name);

    /**
     * @brief Check if the shader exists
     *
     * @param name The name
     * @return True if the shader exists in the shader manager
     */
    static bool hasShader(const std::string& name);

    inline static void destroy() { delete s_instance; }

private:
    static RaytracingShaderManager* s_instance;

    std::unordered_map<std::string, atcg::ref_ptr<RaytracingShader>> _shader;
};
}    // namespace atcg