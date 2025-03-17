#include <Renderer/ShaderCompiler.h>

#include <regex>

#include <glad/glad.h>

namespace atcg
{
ShaderCompiler::ShaderCompiler() {}

ShaderCompiler::~ShaderCompiler() {}

uint32_t ShaderCompiler::compileShader(const std::string& compute_path)
{
    std::string compute_buffer = readShaderCode(compute_path, ShaderType::COMPUTE);
    const char* cShaderCode    = compute_buffer.c_str();

    // Compiling
    uint32_t compute;

    compute = compileShader(GL_COMPUTE_SHADER, cShaderCode);

    // Linking
    uint32_t shaders[] = {compute};
    uint32_t ID        = linkShader(shaders, 1);

    glDeleteShader(compute);

    return ID;
}

uint32_t ShaderCompiler::compileShader(const std::string& vertex_path, const std::string& fragment_path)
{
    // File reading
    std::string vertex_buffer = readShaderCode(vertex_path, ShaderType::VERTEX);
    const char* vShaderCode   = vertex_buffer.c_str();

    std::string fragment_buffer = readShaderCode(fragment_path, ShaderType::FRAGMENT);
    const char* fShaderCode     = fragment_buffer.c_str();

    // Compiling
    uint32_t vertex, fragment;

    vertex   = compileShader(GL_VERTEX_SHADER, vShaderCode);
    fragment = compileShader(GL_FRAGMENT_SHADER, fShaderCode);

    // Linking
    uint32_t shaders[] = {vertex, fragment};
    uint32_t ID        = linkShader(shaders, 2);

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return ID;
}

uint32_t ShaderCompiler::compileShader(const std::string& vertex_path,
                                       const std::string& geometry_path,
                                       const std::string& fragment_path)
{
    // File reading
    std::string vertex_buffer = readShaderCode(vertex_path, ShaderType::VERTEX);
    const char* vShaderCode   = vertex_buffer.c_str();

    std::string fragment_buffer = readShaderCode(fragment_path, ShaderType::FRAGMENT);
    const char* fShaderCode     = fragment_buffer.c_str();

    std::string geometry_buffer = readShaderCode(geometry_path, ShaderType::GEOMETRY);
    const char* gShaderCode     = geometry_buffer.c_str();

    // Compiling
    uint32_t vertex, geometry, fragment;

    vertex   = compileShader(GL_VERTEX_SHADER, vShaderCode);
    geometry = compileShader(GL_GEOMETRY_SHADER, gShaderCode);
    fragment = compileShader(GL_FRAGMENT_SHADER, fShaderCode);

    // Linking
    uint32_t shaders[] = {vertex, geometry, fragment};
    uint32_t ID        = linkShader(shaders, 3);

    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glDeleteShader(geometry);

    return ID;
}

std::pair<bool, std::string> ShaderCompiler::parseIncludeLine(const std::string& line)
{
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));

    if(trimmed.empty() || trimmed[0] != '#')
    {
        return std::make_pair(false, "");
    }

    std::regex includePattern(R"(^\s*#\s*include\s+\"([^\"]+)\")");
    std::smatch match;

    if(std::regex_search(trimmed, match, includePattern))
    {
        return std::make_pair(true, match[1].str());
    }
    return std::make_pair(false, "");
}

std::tuple<bool, std::string, std::string> ShaderCompiler::parseSubroutineUniform(const std::string& line)
{
    std::regex pattern(R"(subroutine\s+uniform\s+(\w+)\s+(\w+);)");
    std::smatch match;

    if(std::regex_search(line, match, pattern))
    {
        return std::make_tuple(true, match[1], match[2]);
    }
    return std::make_tuple(false, "", "");
}

std::string ShaderCompiler::readShaderCode(const std::string& path, const ShaderType type)
{
    std::ifstream shaderFile;

    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    std::stringstream shaderStream;
    try
    {
        shaderFile.open(path);

        shaderStream << shaderFile.rdbuf();

        shaderFile.close();
    }
    catch(std::ifstream::failure e)
    {
        ATCG_ERROR("Could not read shader file: {0}", path);
    }

    std::stringstream parsed_source;

    std::string line;
    while(std::getline(shaderStream, line))
    {
        auto [is_include, include_path]                     = parseIncludeLine(line);
        auto [is_subr_uniform, function_name, uniform_name] = parseSubroutineUniform(line);

        if(is_include)
        {
            std::filesystem::path file_path = path;
            std::string full_include_path   = (file_path.parent_path() / include_path).generic_string();
            std::string included_source     = readShaderCode(full_include_path, type);

            parsed_source << included_source;
        }
        else
        {
            parsed_source << line << "\n";
        }

        if(is_subr_uniform)
        {
            std::unordered_map<std::string, std::string>* subroutines = nullptr;
            switch(type)
            {
                case ShaderType::VERTEX:
                {
                    subroutines = &_vertex_subroutines;
                }
                break;
                case ShaderType::FRAGMENT:
                {
                    subroutines = &_fragment_subroutines;
                }
                break;
                case ShaderType::GEOMETRY:
                {
                    subroutines = &_geometry_subroutines;
                }
                break;
                default:
                {
                    ATCG_ERROR("Shader: Tried to register the subroutine `{0}` with invalid shader type",
                               function_name);
                    continue;
                }
                break;
            }


            (*subroutines)[function_name] = uniform_name;
        }
    }

    return parsed_source.str();
}

uint32_t ShaderCompiler::compileShader(unsigned int shaderType, const std::string& shader_source)
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

uint32_t ShaderCompiler::linkShader(const uint32_t* shaders, const uint32_t& num_shaders)
{
    int32_t success;

    uint32_t ID = glCreateProgram();
    for(uint32_t i = 0; i < num_shaders; ++i)
    {
        glAttachShader(ID, shaders[i]);
    }
    glLinkProgram(ID);

    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if(!success)
    {
        int32_t length;
        glGetProgramiv(ID, GL_INFO_LOG_LENGTH, &length);
        char* infoLog = (char*)malloc(sizeof(char) * length);
        glGetProgramInfoLog(ID, length, &length, infoLog);
        ATCG_ERROR("ERROR::SHADER::PROGRAM::LINKING_FAILED\n{0}", std::string(infoLog));
        free(infoLog);
    }

    return ID;
}
}    // namespace atcg