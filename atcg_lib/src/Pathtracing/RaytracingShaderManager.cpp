#include <Pathtracing/RaytracingShaderManager.h>
#include <Pathtracing/RaytracingShader.h>

namespace atcg
{
RaytracingShaderManager* RaytracingShaderManager::s_instance = new RaytracingShaderManager;


void RaytracingShaderManager::addShader(const std::string& name, const atcg::ref_ptr<RaytracingShader>& shader)
{
    s_instance->_shader.insert(std::make_pair(name, shader));
}

const atcg::ref_ptr<RaytracingShader>& RaytracingShaderManager::getShader(const std::string& name)
{
    return s_instance->_shader[name];
}

bool RaytracingShaderManager::hasShader(const std::string& name)
{
    return s_instance->_shader.find(name) != s_instance->_shader.end();
}
}    // namespace atcg