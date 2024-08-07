#pragma once

#include <variant>
#include <unordered_map>

#include <torch/types.h>

#include <Core/glm.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
/**
 * @brief A class to model a raytracing shader
 */
class RaytracingShader
{
public:
    /**
     * @brief Destructor
     */
    virtual ~RaytracingShader() {}

    /**
     * @brief Set the scene
     *
     * @param scene The scene
     */
    virtual void setScene(const atcg::ref_ptr<Scene>& scene) { _scene = scene; }

    /**
     * @brief Set the camera
     *
     * @param camera The camera
     */
    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) { _camera = camera; };

    /**
     * @brief Set an int
     *
     * @param name The name
     * @param value The value
     */
    void setInt(const std::string& name, const int value);

    /**
     * @brief Set a float
     *
     * @param name The name
     * @param value The value
     */
    void setFloat(const std::string& name, const float value);

    /**
     * @brief Set a vec2
     *
     * @param name The name
     * @param value The value
     */
    void setVec2(const std::string& name, const glm::vec2& value);

    /**
     * @brief Set a vec3
     *
     * @param name The name
     * @param value The value
     */
    void setVec3(const std::string& name, const glm::vec3& value);

    /**
     * @brief Set a vec4
     *
     * @param name The name
     * @param value The value
     */
    void setVec4(const std::string& name, const glm::vec4& value);

    /**
     * @brief Set a mat3
     *
     * @param name The name
     * @param value The value
     */
    void setMat3(const std::string& name, const glm::mat3& value);

    /**
     * @brief Set a mat4
     *
     * @param name The name
     * @param value The value
     */
    void setMat4(const std::string& name, const glm::mat4& value);

    /**
     * @brief Set a tensor
     *
     * @param name The name
     * @param value The value
     */
    void setTensor(const std::string& name, const torch::Tensor& value);

    /**
     * @brief Get an int
     *
     * @param name The name
     *
     * @return The value
     */
    int getInt(const std::string& name) const;

    /**
     * @brief Get a float
     *
     * @param name The name
     *
     * @return The value
     */
    float getFloat(const std::string& name) const;

    /**
     * @brief Get a vec2
     *
     * @param name The name
     *
     * @return The value
     */
    glm::vec2 getVec2(const std::string& name) const;

    /**
     * @brief Get a vec3
     *
     * @param name The name
     *
     * @return The value
     */
    glm::vec3 getVec3(const std::string& name) const;

    /**
     * @brief Get a vec4
     *
     * @param name The name
     *
     * @return The value
     */
    glm::vec4 getVec4(const std::string& name) const;

    /**
     * @brief Get a mat3
     *
     * @param name The name
     *
     * @return The value
     */
    glm::mat3 getMat3(const std::string& name) const;

    /**
     * @brief Get a mat4
     *
     * @param name The name
     *
     * @return The value
     */
    glm::mat4 getMat4(const std::string& name) const;

    /**
     * @brief Get a tensor
     *
     * @param name The name
     *
     * @return The value
     */
    torch::Tensor getTensor(const std::string& name) const;

protected:
    using Parameter = std::variant<int, float, glm::vec2, glm::vec3, glm::vec4, glm::mat3, glm::mat4, torch::Tensor>;

    void set(const std::string& name, const Parameter& value);

    std::unordered_map<std::string, Parameter> _parameters;

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<PerspectiveCamera> _camera;
};
}    // namespace atcg