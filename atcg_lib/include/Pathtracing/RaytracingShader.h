#pragma once

#include <variant>
#include <unordered_map>

#include <torch/types.h>

#include <Core/glm.h>
#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
class RaytracingShader
{
public:
    virtual ~RaytracingShader() {}

    virtual void setScene(const atcg::ref_ptr<Scene>& scene) { _scene = scene; }

    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) { _camera = camera; };

    void setInt(const std::string& name, const int value);

    void setFloat(const std::string& name, const float value);

    void setVec2(const std::string& name, const glm::vec2& value);

    void setVec3(const std::string& name, const glm::vec3& value);

    void setVec4(const std::string& name, const glm::vec4& value);

    void setMat3(const std::string& name, const glm::mat3& value);

    void setMat4(const std::string& name, const glm::mat4& value);

    void setTensor(const std::string& name, const torch::Tensor& value);

    int getInt(const std::string& name) const;

    float getFloat(const std::string& name) const;

    glm::vec2 getVec2(const std::string& name) const;

    glm::vec3 getVec3(const std::string& name) const;

    glm::vec4 getVec4(const std::string& name) const;

    glm::mat3 getMat3(const std::string& name) const;

    glm::mat4 getMat4(const std::string& name) const;

    torch::Tensor getTensor(const std::string& name) const;

protected:
    using Parameter = std::variant<int, float, glm::vec2, glm::vec3, glm::vec4, glm::mat3, glm::mat4, torch::Tensor>;

    void set(const std::string& name, const Parameter& value);

    std::unordered_map<std::string, Parameter> _parameters;

    atcg::ref_ptr<Scene> _scene;
    atcg::ref_ptr<PerspectiveCamera> _camera;
};
}    // namespace atcg