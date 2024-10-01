#include <Renderer/ShaderManager.h>
#include <Renderer/Shader.h>
#include <Core/Path.h>

namespace atcg
{

ShaderManagerSystem::ShaderManagerSystem() : _shader_path(atcg::shader_directory()) {}

void ShaderManagerSystem::addShader(const std::string& name, const atcg::ref_ptr<Shader>& shader)
{
    _shader.insert(std::make_pair(name, shader));

    if(shader->isComputeShader())
    {
        const std::string& compute_path = shader->getComputePath();
        _time_stamps.insert(std::make_pair(compute_path, std::filesystem::last_write_time(compute_path)));
    }
    else
    {
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
}

void ShaderManagerSystem::addShaderFromName(const std::string& name)
{
    std::string vertex_path   = (_shader_path / (name + ".vs")).string();
    std::string fragment_path = (_shader_path / (name + ".fs")).string();
    std::string geometry_path = (_shader_path / (name + ".gs")).string();

    if(!std::filesystem::exists(vertex_path) || !std::filesystem::exists(fragment_path))
    {
        ATCG_ERROR("Shader: {0} needs at least a vertex and fragment shader!", name);
        return;
    }

    atcg::ref_ptr<Shader> shader;
    if(std::filesystem::exists(geometry_path))
    {
        shader = atcg::make_ref<Shader>(vertex_path, fragment_path, geometry_path);
    }
    else
    {
        shader = atcg::make_ref<Shader>(vertex_path, fragment_path);
    }

    addShader(name, shader);
}

void ShaderManagerSystem::addComputeShaderFromName(const std::string& name)
{
    std::string compute_path = (_shader_path / (name + ".glsl")).string();

    if(!std::filesystem::exists(compute_path))
    {
        ATCG_ERROR("Shader: {0} cannot be found!", name);
        return;
    }

    atcg::ref_ptr<Shader> shader = atcg::make_ref<Shader>(compute_path);

    addShader(name, shader);
}

bool ShaderManagerSystem::hasShader(const std::string& name)
{
    return _shader.find(name) != _shader.end();
}

const atcg::ref_ptr<Shader>& ShaderManagerSystem::getShader(const std::string& name)
{
    return _shader[name];
}

void ShaderManagerSystem::onUpdate()
{
    for(auto& shader: _shader)
    {
        if(shader.second->isComputeShader())
        {
            const std::string& compute_path = shader.second->getComputePath();
            bool recompile                  = false;

            auto time_stamp_cs = std::filesystem::last_write_time(compute_path);
            if(_time_stamps[compute_path] != time_stamp_cs)
            {
                _time_stamps[compute_path] = time_stamp_cs;

                shader.second->recompile(compute_path);
            }
        }
        else
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
                if(has_geoemtry)
                    shader.second->recompile(vertex_path, fragment_path, geometry_path);
                else
                    shader.second->recompile(vertex_path, fragment_path);
            }
        }
    }
}
}    // namespace atcg