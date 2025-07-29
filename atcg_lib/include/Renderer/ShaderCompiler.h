#pragma once

#include <string>
#include <cstdint>

#include <Renderer/ShaderType.h>

namespace atcg
{

class Shader;

/**
 * @brief A class to model a shader compiler.
 * This class is used internally to create shader. Calling this class manually is not advised.
 */
class ShaderCompiler
{
public:
    /**
     * @brief Constructor
     */
    ShaderCompiler();

    /**
     * @brief Destructor
     */
    ~ShaderCompiler();

    /**
     * @brief Compile a geoemtry shader
     *
     * @param compute_path The path to the compute shader
     *
     * @return The shader ID
     */
    uint32_t compileShader(const std::string& compute_path);

    /**
     * @brief Compile a standard shader
     *
     * @param vertex_path The path to the vertex shader
     * @param fragment_path The path to the fragment shader
     *
     * @return The shader ID
     */
    uint32_t compileShader(const std::string& vertex_path, const std::string& fragment_path);

    /**
     * @brief Compile a shader with a geometry shader
     *
     * @param vertex_path The path to the vertex shader
     * @param geometry_path The path to the geometry shader
     * @param fragment_path The path to the fragment shader
     *
     * @return The shader ID
     */
    uint32_t
    compileShader(const std::string& vertex_path, const std::string& geometry_path, const std::string& fragment_path);

    /**
     * @brief Get the (parsed) source code of a shader
     *
     * @param type The shader type
     *
     * @return The parsed source code
     */
    std::string getSource(const ShaderType& type) const;

private:
    friend class Shader;

    std::pair<bool, std::string> parseIncludeLine(const std::string& line);

    std::tuple<bool, std::string, std::string> parseSubroutineUniform(const std::string& line);

    std::string readShaderCode(const std::string& path, const ShaderType type);

    uint32_t compileShader(unsigned int shaderType, const std::string& shader_source);

    uint32_t linkShader(const uint32_t* shaders, const uint32_t& num_shaders);

private:
    std::unordered_map<std::string, std::string> _vertex_subroutines;
    std::unordered_map<std::string, std::string> _fragment_subroutines;
    std::unordered_map<std::string, std::string> _geometry_subroutines;

    std::string _vertex_code   = "";
    std::string _fragment_code = "";
    std::string _geometry_code = "";
    std::string _compute_code  = "";
};
}    // namespace atcg