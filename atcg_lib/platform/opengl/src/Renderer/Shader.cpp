#include <Renderer/Shader.h>
#include <Renderer/ShaderCompiler.h>
#include <glad/glad.h>

namespace atcg
{

namespace detail
{
static GLenum shaderToGL(const ShaderType type)
{
    switch(type)
    {
        case ShaderType::VERTEX:
        {
            return GL_VERTEX_SHADER;
        }
        break;
        case ShaderType::FRAGMENT:
        {
            return GL_FRAGMENT_SHADER;
        }
        break;
        case ShaderType::COMPUTE:
        {
            return GL_COMPUTE_SHADER;
        }
        break;
        case ShaderType::GEOMETRY:
        {
            return GL_GEOMETRY_SHADER;
        }
        break;
    }
    return 0;
}
}    // namespace detail

Shader::Shader(const std::string& compute_path)
{
    recompile(compute_path);
}

Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath)
{
    recompile(vertexPath, fragmentPath);
}

Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath)
{
    recompile(vertexPath, fragmentPath, geometryPath);
}

Shader::~Shader()
{
    glDeleteProgram(_ID);
    _ID = 0;
}

void Shader::recompile(const std::string& compute_path)
{
    if(_ID != 0)
    {
        glDeleteProgram(_ID);
        _ID = 0;
    }

    ShaderCompiler compiler;
    _ID = compiler.compileShader(compute_path);

    _has_geometry = false;
    _is_compute   = true;
    _compute_path = compute_path;

    // Update uniform locations
    for(auto it = _uniforms.begin(); it != _uniforms.end(); ++it)
    {
        it->second.location = glGetUniformLocation(_ID, it->first.c_str());
    }
}

void Shader::recompile(const std::string& vertex_path, const std::string& fragment_path)
{
    if(_ID != 0)
    {
        glDeleteProgram(_ID);
        _ID = 0;
    }

    ShaderCompiler compiler;
    _ID = compiler.compileShader(vertex_path, fragment_path);

    _has_geometry  = false;
    _is_compute    = false;
    _vertex_path   = vertex_path;
    _fragment_path = fragment_path;

    // Update uniform locations
    for(auto it = _uniforms.begin(); it != _uniforms.end(); ++it)
    {
        it->second.location = glGetUniformLocation(_ID, it->first.c_str());
    }
}

void Shader::recompile(const std::string& vertex_path,
                       const std::string& fragment_path,
                       const std::string& geometry_path)
{
    if(_ID != 0)
    {
        glDeleteProgram(_ID);
        _ID = 0;
    }

    ShaderCompiler compiler;
    _ID = compiler.compileShader(vertex_path, geometry_path, fragment_path);

    _has_geometry  = true;
    _is_compute    = false;
    _vertex_path   = vertex_path;
    _geometry_path = geometry_path;
    _fragment_path = fragment_path;

    // Update uniform locations
    for(auto it = _uniforms.begin(); it != _uniforms.end(); ++it)
    {
        it->second.location = glGetUniformLocation(_ID, it->first.c_str());
    }
}

Shader::Uniform& Shader::getUniform(const std::string& name)
{
    auto it = _uniforms.find(name);
    if(it == _uniforms.end())
    {
        Uniform uniform;
        uniform.location = glGetUniformLocation(_ID, name.c_str());
        _uniforms.insert(std::make_pair(name, uniform));
        it = _uniforms.find(name);
    }
    return it->second;
}

template<typename T>
void Shader::setValue(const uint32_t location, const T& value) const
{
    throw std::invalid_argument("Shader Set not implemented for this datatype!");
}

template<>
void Shader::setValue<int>(const uint32_t location, const int& value) const
{
    glUniform1i(location, value);
}

template<>
void Shader::setValue<float>(const uint32_t location, const float& value) const
{
    glUniform1f(location, value);
}

template<>
void Shader::setValue<glm::vec2>(const uint32_t location, const glm::vec2& value) const
{
    glUniform2f(location, value.x, value.y);
}

template<>
void Shader::setValue<glm::vec3>(const uint32_t location, const glm::vec3& value) const
{
    glUniform3f(location, value.x, value.y, value.z);
}

template<>
void Shader::setValue<glm::vec4>(const uint32_t location, const glm::vec4& value) const
{
    glUniform4f(location, value.x, value.y, value.z, value.w);
}

template<>
void Shader::setValue<glm::mat4>(const uint32_t location, const glm::mat4& value) const
{
    glUniformMatrix4fv(location, 1, GL_FALSE, &value[0][0]);
}

void Shader::setInt(const std::string& name, const int& value)
{
    Uniform& uniform = getUniform(name);
    uniform.type     = ShaderDataType::Int;
    uniform.data     = value;
}

void Shader::setFloat(const std::string& name, const float& value)
{
    Uniform& uniform = getUniform(name);
    uniform.type     = ShaderDataType::Float;
    uniform.data     = value;
}

void Shader::setVec2(const std::string& name, const glm::vec2& value)
{
    Uniform& uniform = getUniform(name);
    uniform.type     = ShaderDataType::Float2;
    uniform.data     = value;
}

void Shader::setVec3(const std::string& name, const glm::vec3& value)
{
    Uniform& uniform = getUniform(name);
    uniform.type     = ShaderDataType::Float3;
    uniform.data     = value;
}

void Shader::setVec4(const std::string& name, const glm::vec4& value)
{
    Uniform& uniform = getUniform(name);
    uniform.type     = ShaderDataType::Float4;
    uniform.data     = value;
}

void Shader::setMat4(const std::string& name, const glm::mat4& value)
{
    Uniform& uniform = getUniform(name);
    uniform.type     = ShaderDataType::Mat4;
    uniform.data     = value;
}

void Shader::registerSubroutine(const std::string& subroutine_type, const ShaderType type)
{
    SubroutineInfo& subroutine = _subroutines[subroutine_type];
    subroutine.type            = type;
    switch(type)
    {
        case ShaderType::VERTEX:
        {
            subroutine.subroutine_index = _vertex_subroutines.size();
            _vertex_subroutines.resize(subroutine.subroutine_index + 1);
        }
        break;
        case ShaderType::FRAGMENT:
        {
            subroutine.subroutine_index = _fragment_subroutines.size();
            _fragment_subroutines.resize(subroutine.subroutine_index + 1);
        }
        break;
        case ShaderType::GEOMETRY:
        {
            subroutine.subroutine_index = _geometry_subroutines.size();
            _geometry_subroutines.resize(subroutine.subroutine_index + 1);
        }
        break;
        default:
        {
            ATCG_ERROR("Shader: Tried to register a subroutine `{0}` to invalid shader type: `{1}`",
                       subroutine_type,
                       (int)type);
        }
        break;
    }
}

void Shader::selectSubroutine(const std::string& subroutine_type, const std::string& subroutine_name)
{
    auto it = _subroutines.find(subroutine_type);
    if(it == _subroutines.end())
    {
        ATCG_WARN("SHADER: Tried to select subroutine `{0}` of non-registered type `{1}`. Call "
                  "Shader::registerSubroutine "
                  "first.",
                  subroutine_name,
                  subroutine_type);
        return;
    }

    std::vector<uint32_t>* indices = nullptr;
    switch(it->second.type)
    {
        case ShaderType::VERTEX:
        {
            indices = &_vertex_subroutines;
        }
        break;
        case ShaderType::FRAGMENT:
        {
            indices = &_fragment_subroutines;
        }
        break;
        case ShaderType::GEOMETRY:
        {
            indices = &_geometry_subroutines;
        }
        break;
        default:
        {
            ATCG_ERROR("Shader: Tried to select the subroutine `{0}` of type `{1}` with invalid shader type",
                       subroutine_name,
                       subroutine_type);
            return;
        }
        break;
    }

    (*indices)[it->second.subroutine_index] =
        glGetSubroutineIndex(_ID, detail::shaderToGL(it->second.type), subroutine_name.c_str());
}

void Shader::setMVP(const glm::mat4& M, const glm::mat4& V, const glm::mat4& P)
{
    setMat4("M", M);
    setMat4("V", V);
    setMat4("P", P);
}

void Shader::use() const
{
    glUseProgram(_ID);
    for(auto it = _uniforms.begin(); it != _uniforms.end(); ++it)
    {
        const Uniform& uniform = it->second;
        switch(uniform.type)
        {
            case ShaderDataType::Int:
            case ShaderDataType::Bool:
            {
                setValue<int>(uniform.location, std::get<int>(uniform.data));
            }
            break;
            case ShaderDataType::Float:
            {
                setValue<float>(uniform.location, std::get<float>(uniform.data));
            }
            break;
            case ShaderDataType::Float2:
            {
                setValue<glm::vec2>(uniform.location, std::get<glm::vec2>(uniform.data));
            }
            break;
            case ShaderDataType::Float3:
            {
                setValue<glm::vec3>(uniform.location, std::get<glm::vec3>(uniform.data));
            }
            break;
            case ShaderDataType::Float4:
            {
                setValue<glm::vec4>(uniform.location, std::get<glm::vec4>(uniform.data));
            }
            break;
            case ShaderDataType::Mat4:
            {
                setValue<glm::mat4>(uniform.location, std::get<glm::mat4>(uniform.data));
            }
            break;
        }
    }

    // Subroutines
    if(!_vertex_subroutines.empty())
        glUniformSubroutinesuiv(GL_VERTEX_SHADER, _vertex_subroutines.size(), _vertex_subroutines.data());
    if(!_fragment_subroutines.empty())
        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, _fragment_subroutines.size(), _fragment_subroutines.data());
    if(!_geometry_subroutines.empty())
        glUniformSubroutinesuiv(GL_GEOMETRY_SHADER, _geometry_subroutines.size(), _geometry_subroutines.data());
}

void Shader::dispatch(const glm::ivec3& work_groups) const
{
    use();
    glDispatchCompute(work_groups.x, work_groups.y, work_groups.z);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

}    // namespace atcg