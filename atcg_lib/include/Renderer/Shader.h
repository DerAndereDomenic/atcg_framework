#pragma once

#include <string>
#include <glm/glm.hpp>

namespace atcg
{
/**
 * @brief This class models a shader
 */
class Shader
{
public:
    /**
     * @brief Construct a new Shader object
     */
    Shader() = default;

    /**
     * @brief Construct a new Shader object.
     * This will be a compute shader outside of the standard graphics pipeline
     *
     * @param compute_path The path to the compute shader file
     */
    Shader(const std::string& compute_path);

    /**
     * @brief Construct a new Shader object
     *
     * @param vertex_path The path to the vertex shader
     * @param fragment_path The path to the fragment shader
     */
    Shader(const std::string& vertex_path, const std::string& fragment_path);

    /**
     * @brief Construct a new Shader object
     *
     * @param vertex_path The path to the vertex shader
     * @param fragment_path The path to the fragment shader
     * @param geometry_shader The path to the geometry shader
     */
    Shader(const std::string& vertex_path, const std::string& fragment_path, const std::string& geometry_shader);

    /**
     * @brief Destroy the Shader object
     */
    ~Shader();

    /**
     * @brief Use the shader
     */
    void use() const;

    /**
     * @brief Set an int uniform
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setInt(const std::string& name, const int& value);

    /**
     * @brief Set a float uniform
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setFloat(const std::string& name, const float& value);

    /**
     * @brief Set a vec2 uniform
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setVec2(const std::string& name, const glm::vec2& value);

    /**
     * @brief Set a vec3 uniform
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setVec3(const std::string& name, const glm::vec3& value);

    /**
     * @brief Set a vec4 uniform
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setVec4(const std::string& name, const glm::vec4& value);

    /**
     * @brief Set a mat4 uniform
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setMat4(const std::string& name, const glm::mat4& value);

    /**
     * @brief Set the model, view, and projection matrix
     *
     * @param M The model matrix
     * @param V The view matrix
     * @param P The projection matrix
     */
    void
    setMVP(const glm::mat4& M = glm::mat4(1), const glm::mat4& V = glm::mat4(1), const glm::mat4& P = glm::mat4(1));

    /**
     * @brief Dispatch the shader if it is a compute shader
     *
     * @param work_groups The number of work groups in each dimension
     */
    void dispatch(const glm::ivec3& work_groups) const;

    inline bool hasGeometryShader() const { return _has_geometry; }

    inline bool isComputeShader() const { return _is_compute; }

    inline const std::string& getVertexPath() const { return _vertex_path; }

    inline const std::string& getGeometryPath() const { return _geometry_path; }

    inline const std::string& getFragmentPath() const { return _fragment_path; }

    inline const std::string& getComputePath() const { return _compute_path; }

private:
    void readShaderCode(const std::string& path, std::string* code);

    uint32_t compileShader(unsigned int shaderType, const std::string& shader_source);

    void linkShader(const uint32_t* shaders, const uint32_t& num_shaders);

private:
    uint32_t _ID               = 0;
    std::string _vertex_path   = "";
    std::string _fragment_path = "";
    std::string _geometry_path = "";
    std::string _compute_path  = "";
    bool _has_geometry         = false;
    bool _is_compute           = false;
};
}    // namespace atcg