#include <Renderer/Shader.h>
#include <glad/glad.h>

namespace atcg
{
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
{    // File reading
    std::string compute_buffer;

    readShaderCode(compute_path, &compute_buffer);
    const char* cShaderCode = compute_buffer.c_str();

    // Compiling
    uint32_t compute;

    compute = compileShader(GL_COMPUTE_SHADER, cShaderCode);

    // Linking
    uint32_t shaders[] = {compute};
    linkShader(shaders, 1);

    glDeleteShader(compute);

    _has_geometry = false;
    _is_compute   = true;
    _compute_path = compute_path;
}

void Shader::recompile(const std::string& vertex_path, const std::string& fragment_path)
{
    if(_ID != 0)
    {
        glDeleteProgram(_ID);
        _ID = 0;
    }

    // File reading
    std::string vertex_buffer, fragment_buffer;

    readShaderCode(vertex_path, &vertex_buffer);
    const char* vShaderCode = vertex_buffer.c_str();

    readShaderCode(fragment_path, &fragment_buffer);
    const char* fShaderCode = fragment_buffer.c_str();

    // Compiling
    uint32_t vertex, fragment;

    vertex   = compileShader(GL_VERTEX_SHADER, vShaderCode);
    fragment = compileShader(GL_FRAGMENT_SHADER, fShaderCode);

    // Linking
    uint32_t shaders[] = {vertex, fragment};
    linkShader(shaders, 2);

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    _has_geometry  = false;
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

    // File reading
    std::string vertex_buffer, fragment_buffer, geometry_buffer;

    readShaderCode(vertex_path, &vertex_buffer);
    const char* vShaderCode = vertex_buffer.c_str();

    readShaderCode(fragment_path, &fragment_buffer);
    const char* fShaderCode = fragment_buffer.c_str();

    readShaderCode(geometry_path, &geometry_buffer);
    const char* gShaderCode = geometry_buffer.c_str();

    // Compiling
    uint32_t vertex, geometry, fragment;

    vertex   = compileShader(GL_VERTEX_SHADER, vShaderCode);
    geometry = compileShader(GL_GEOMETRY_SHADER, gShaderCode);
    fragment = compileShader(GL_FRAGMENT_SHADER, fShaderCode);

    // Linking
    uint32_t shaders[] = {vertex, geometry, fragment};
    linkShader(shaders, 3);

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glDeleteShader(geometry);

    _has_geometry  = true;
    _vertex_path   = vertex_path;
    _geometry_path = geometry_path;
    _fragment_path = fragment_path;

    // Update uniform locations
    for(auto it = _uniforms.begin(); it != _uniforms.end(); ++it)
    {
        it->second.location = glGetUniformLocation(_ID, it->first.c_str());
    }
}

void Shader::readShaderCode(const std::string& path, std::string* code)
{
    std::ifstream shaderFile;

    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        shaderFile.open(path);
        std::stringstream shaderStream;

        shaderStream << shaderFile.rdbuf();

        shaderFile.close();

        *code = shaderStream.str();
    }
    catch(std::ifstream::failure e)
    {
        ATCG_ERROR("Could not read shader file: {0}", path);
    }
}

uint32_t Shader::compileShader(unsigned int shaderType, const std::string& shader_source)
{
    uint32_t shader;
    int32_t success;

    shader                          = glCreateShader(shaderType);
    const char* shader_source_c_str = shader_source.c_str();
    glShaderSource(shader, 1, &shader_source_c_str, NULL);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if(!success)
    {
        int32_t length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        char* infoLog = (char*)malloc(sizeof(char) * length);
        glGetShaderInfoLog(shader, length, &length, infoLog);
        if(shaderType == GL_VERTEX_SHADER)
        {
            ATCG_ERROR("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{0}", std::string(infoLog));
        }
        else if(shaderType == GL_FRAGMENT_SHADER)
        {
            ATCG_ERROR("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n{0}", std::string(infoLog));
        }
        else if(shaderType == GL_GEOMETRY_SHADER)
        {
            ATCG_ERROR("ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n{0}", std::string(infoLog));
        }
        else if(shaderType == GL_COMPUTE_SHADER)
        {
            ATCG_ERROR("ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n{0}", std::string(infoLog));
        }
        else
        {
            ATCG_ERROR("ERROR::SHADER::COMPILATION_FAILED\nUnknown shader type");
        }

        free(infoLog);
    }

    return shader;
}

void Shader::linkShader(const uint32_t* shaders, const uint32_t& num_shaders)
{
    int32_t success;

    _ID = glCreateProgram();
    for(uint32_t i = 0; i < num_shaders; ++i)
    {
        glAttachShader(_ID, shaders[i]);
    }
    glLinkProgram(_ID);

    glGetProgramiv(_ID, GL_LINK_STATUS, &success);
    if(!success)
    {
        int32_t length;
        glGetProgramiv(_ID, GL_INFO_LOG_LENGTH, &length);
        char* infoLog = (char*)malloc(sizeof(char) * length);
        glGetProgramInfoLog(_ID, length, &length, infoLog);
        ATCG_ERROR("ERROR::SHADER::PROGRAM::LINKING_FAILED\n{0}", std::string(infoLog));
        free(infoLog);
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
}

void Shader::dispatch(const glm::ivec3& work_groups) const
{
    use();
    glDispatchCompute(work_groups.x, work_groups.y, work_groups.z);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

}    // namespace atcg