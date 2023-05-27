#include <Renderer/ShaderManager.h>

#include <iostream>

namespace atcg
{
ShaderManager* ShaderManager::s_instance = new ShaderManager;

void ShaderManager::addShaderImpl(const std::string& name, const atcg::ref_ptr<Shader>& shader)
{
    _shader.insert(std::make_pair(name, shader));
    const std::string& vertex_path   = shader->getVertexPath();
    const std::string& fragment_path = shader->getFragmentPath();

    _time_stamps.insert(std::make_pair(vertex_path, std::filesystem::last_write_time(vertex_path)));
    _time_stamps.insert(std::make_pair(fragment_path, std::filesystem::last_write_time(fragment_path)));

    if(shader->hasGeometryShader())
    {
        const std::string& geometry_path = shader->getGeometryPath();
        _time_stamps.insert(std::make_pair(geometry_path, std::filesystem::last_write_time(geometry_path)));
    }
}

void ShaderManager::addShaderFromNameImpl(const std::string& name)
{
    std::string vertex_path   = "shader/" + name + ".vs";
    std::string fragment_path = "shader/" + name + ".fs";
    std::string geometry_path = "shader/" + name + ".gs";

    if(!std::filesystem::exists(vertex_path) || !std::filesystem::exists(fragment_path))
    {
        std::cerr << "Shader: " << name << " needs at least a vertex and fragment shader\n";
        return;
    }

    atcg::ref_ptr<Shader> shader;
    if(std::filesystem::exists(geometry_path))
    {
        shader = atcg::make_ref<Shader>(vertex_path, geometry_path, fragment_path);
    }
    else { shader = atcg::make_ref<Shader>(vertex_path, fragment_path); }

    addShaderImpl(name, shader);
}

const atcg::ref_ptr<Shader>& ShaderManager::getShaderImpl(const std::string& name)
{
    return _shader[name];
}

void ShaderManager::onUpdateImpl()
{
    for(auto& shader: _shader)
    {
        const std::string& vertex_path   = shader.second->getVertexPath();
        const std::string& fragment_path = shader.second->getFragmentPath();
        const std::string& geometry_path = shader.second->getGeometryPath();
        bool has_geoemtry                = shader.second->hasGeometryShader();
        bool recompile                   = false;

        auto time_stamp_vs = std::filesystem::last_write_time(vertex_path);
        if(_time_stamps[vertex_path] != time_stamp_vs)
        {
            _time_stamps[vertex_path] = time_stamp_vs;
            recompile                 = true;
        }

        auto time_stamp_fs = std::filesystem::last_write_time(fragment_path);
        if(_time_stamps[fragment_path] != time_stamp_fs)
        {
            _time_stamps[fragment_path] = time_stamp_fs;
            recompile                   = true;
        }

        if(has_geoemtry)
        {
            auto time_stamp_gs = std::filesystem::last_write_time(geometry_path);
            if(_time_stamps[geometry_path] != time_stamp_gs)
            {
                _time_stamps[geometry_path] = time_stamp_gs;
                recompile                   = true;
            }
        }

        if(recompile)
        {
            atcg::host_allocator allocator;
            Shader* shader_buffer = static_cast<Shader*>(allocator.allocate(sizeof(Shader)));
            if(has_geoemtry)
                shader.second.reset(new(shader_buffer) Shader(vertex_path, geometry_path, fragment_path));
            else
                shader.second.reset(new(shader_buffer) Shader(vertex_path, fragment_path));
        }
    }
}
}    // namespace atcg