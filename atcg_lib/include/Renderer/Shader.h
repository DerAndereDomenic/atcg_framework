#pragma once

#include <Core/glm.h>
#include <Renderer/Buffer.h>

#include <string>
#include <variant>
#include <unordered_map>

namespace atcg
{

enum class ShaderType
{
    None = 0,
    VERTEX,
    FRAGMENT,
    GEOMETRY,
    COMPUTE
};

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
     * @param geometry_path The path to the geometry shader
     */
    Shader(const std::string& vertex_path, const std::string& fragment_path, const std::string& geometry_path);

    /**
     * @brief Destroy the Shader object
     */
    ~Shader();

    /**
     * @brief Recompile a compute shader with new source
     *
     * @param compute_path The path to the compute shader
     */
    void recompile(const std::string& compute_path);

    /**
     * @brief Recompile a shader with new source
     *
     * @param vertex_path The path to the vertex shader
     * @param fragment_path The path to the fragment shader
     */
    void recompile(const std::string& vertex_path, const std::string& fragment_path);

    /**
     * @brief Recompile a shader with new source
     *
     * @param vertex_path The path to the vertex shader
     * @param fragment_path The path to the fragment shader
     * @param geometry_path The path to the geometry shader
     */
    void recompile(const std::string& vertex_path, const std::string& fragment_path, const std::string& geometry_path);

    /**
     * @brief Use the shader.
     * This sets all the shader uniforms so it should always be called last before doing the draw call.
     * Typically the client does not have to use it as every Rendering command uses the shader at some point.
     */
    void use() const;

    /**
     * @brief Set an int uniform.
     * All shader uniforms are uploaded in a deferred way. I.e. this function caches the location and value of the
     * uniform. The upload to the actual shader program is done when calling shader->use();
     * This cache is persistent over frames, so "const" values can be set only once and do not have to be reset every
     * frame.
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setInt(const std::string& name, const int& value);

    /**
     * @brief Set a float uniform.
     * All shader uniforms are uploaded in a deferred way. I.e. this function caches the location and value of the
     * uniform. The upload to the actual shader program is done when calling shader->use();
     * This cache is persistent over frames, so "const" values can be set only once and do not have to be reset every
     * frame.
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setFloat(const std::string& name, const float& value);

    /**
     * @brief Set a vec2 uniform.
     * All shader uniforms are uploaded in a deferred way. I.e. this function caches the location and value of the
     * uniform. The upload to the actual shader program is done when calling shader->use();
     * This cache is persistent over frames, so "const" values can be set only once and do not have to be reset every
     * frame.
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setVec2(const std::string& name, const glm::vec2& value);

    /**
     * @brief Set a vec3 uniform.
     * All shader uniforms are uploaded in a deferred way. I.e. this function caches the location and value of the
     * uniform. The upload to the actual shader program is done when calling shader->use();
     * This cache is persistent over frames, so "const" values can be set only once and do not have to be reset every
     * frame.
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setVec3(const std::string& name, const glm::vec3& value);

    /**
     * @brief Set a vec4 uniform.
     * All shader uniforms are uploaded in a deferred way. I.e. this function caches the location and value of the
     * uniform. The upload to the actual shader program is done when calling shader->use();
     * This cache is persistent over frames, so "const" values can be set only once and do not have to be reset every
     * frame.
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setVec4(const std::string& name, const glm::vec4& value);

    /**
     * @brief Set a mat4 uniform.
     * All shader uniforms are uploaded in a deferred way. I.e. this function caches the location and value of the
     * uniform. The upload to the actual shader program is done when calling shader->use();
     * This cache is persistent over frames, so "const" values can be set only once and do not have to be reset every
     * frame.
     *
     * @param name Name of the uniform
     * @param value The value
     */
    void setMat4(const std::string& name, const glm::mat4& value);

    /**
     * @brief Register a subroutine
     *
     * @param subroutine_type The name of the subroutine definition
     * @param type The shader type where the subroutine is defined
     */
    void registerSubroutine(const std::string& subroutine_type, const ShaderType type);

    /**
     * @brief Choose a subroutine
     *
     * @param subroutine_type The name of the subroutine
     * @param subroutine_name The shader type where the subroutine is located
     */
    void selectSubroutine(const std::string& subroutine_type, const std::string& subroutine_name);

    /**
     * @brief Set the model, view, and projection matrix.
     *
     * @param M The model matrix
     * @param V The view matrix
     * @param P The projection matrix
     */
    void
    setMVP(const glm::mat4& M = glm::mat4(1), const glm::mat4& V = glm::mat4(1), const glm::mat4& P = glm::mat4(1));

    /**
     * @brief Dispatch the shader if it is a compute shader.
     *
     * @param work_groups The number of work groups in each dimension
     */
    void dispatch(const glm::ivec3& work_groups) const;

    ATCG_INLINE bool hasGeometryShader() const { return _has_geometry; }

    ATCG_INLINE bool isComputeShader() const { return _is_compute; }

    ATCG_INLINE const std::string& getVertexPath() const { return _vertex_path; }

    ATCG_INLINE const std::string& getGeometryPath() const { return _geometry_path; }

    ATCG_INLINE const std::string& getFragmentPath() const { return _fragment_path; }

    ATCG_INLINE const std::string& getComputePath() const { return _compute_path; }

private:
    struct Uniform
    {
        uint32_t location;
        ShaderDataType type;
        std::variant<int, float, glm::vec2, glm::vec3, glm::vec4, glm::mat4> data;
    };

    struct SubroutineInfo
    {
        ShaderType type;
        uint32_t subroutine_index;
    };

    Uniform& getUniform(const std::string& name);

    template<typename T>
    void setValue(const uint32_t location, const T& value) const;

private:
    uint32_t _ID               = 0;
    std::string _vertex_path   = "";
    std::string _fragment_path = "";
    std::string _geometry_path = "";
    std::string _compute_path  = "";
    bool _has_geometry         = false;
    bool _is_compute           = false;
    std::unordered_map<std::string, Uniform> _uniforms;
    std::unordered_map<std::string, SubroutineInfo> _subroutines;
    std::vector<uint32_t> _vertex_subroutines;
    std::vector<uint32_t> _fragment_subroutines;
    std::vector<uint32_t> _geometry_subroutines;
};
}    // namespace atcg