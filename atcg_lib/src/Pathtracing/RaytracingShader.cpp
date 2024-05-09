#include <Pathtracing/RaytracingShader.h>

namespace atcg
{
void RaytracingShader::set(const std::string& name, const Parameter& value)
{
    _parameters[name] = value;
}

void RaytracingShader::setInt(const std::string& name, const int value)
{
    set(name, value);
}

void RaytracingShader::setFloat(const std::string& name, const float value)
{
    set(name, value);
}

void RaytracingShader::setVec2(const std::string& name, const glm::vec2& value)
{
    set(name, value);
}

void RaytracingShader::setVec3(const std::string& name, const glm::vec3& value)
{
    set(name, value);
}

void RaytracingShader::setVec4(const std::string& name, const glm::vec4& value)
{
    set(name, value);
}

void RaytracingShader::setMat3(const std::string& name, const glm::mat3& value)
{
    set(name, value);
}

void RaytracingShader::setMat4(const std::string& name, const glm::mat4& value)
{
    set(name, value);
}

void RaytracingShader::setTensor(const std::string& name, const torch::Tensor& value)
{
    set(name, value);
}

int RaytracingShader::getInt(const std::string& name) const
{
    return std::get<int>(_parameters.at(name));
}

float RaytracingShader::getFloat(const std::string& name) const
{
    return std::get<float>(_parameters.at(name));
}

glm::vec2 RaytracingShader::getVec2(const std::string& name) const
{
    return std::get<glm::vec2>(_parameters.at(name));
}

glm::vec3 RaytracingShader::getVec3(const std::string& name) const
{
    return std::get<glm::vec3>(_parameters.at(name));
}

glm::vec4 RaytracingShader::getVec4(const std::string& name) const
{
    return std::get<glm::vec4>(_parameters.at(name));
}

glm::mat3 RaytracingShader::getMat3(const std::string& name) const
{
    return std::get<glm::mat3>(_parameters.at(name));
}

glm::mat4 RaytracingShader::getMat4(const std::string& name) const
{
    return std::get<glm::mat4>(_parameters.at(name));
}

torch::Tensor RaytracingShader::getTensor(const std::string& name) const
{
    return std::get<torch::Tensor>(_parameters.at(name));
}
}    // namespace atcg