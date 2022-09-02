#include <Renderer/Shader.h>
#include <glad/glad.h>
#include <fstream>
#include <sstream>
#include <iostream>

namespace atcg
{
    Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath)
    {
        //File reading
        std::string vertex_buffer, fragment_buffer;

        readShaderCode(vertexPath, &vertex_buffer);
        const char* vShaderCode = vertex_buffer.c_str();

        readShaderCode(fragmentPath, &fragment_buffer);
        const char* fShaderCode = fragment_buffer.c_str();

        //Compiling
        uint32_t vertex, fragment;
        
        vertex = compileShader(GL_VERTEX_SHADER, vShaderCode);
        fragment = compileShader(GL_FRAGMENT_SHADER, fShaderCode);

        //Linking
        uint32_t shaders[] = { vertex,fragment };
        linkShader(shaders, 2);

        glDeleteShader(vertex);
        glDeleteShader(fragment);

        _has_geometry = false;
        _vertex_path = vertexPath;
        _fragment_path = fragmentPath;
    }

    Shader::Shader(const std::string& vertexPath, const std::string& geometryPath, const std::string& fragmentPath)
    {
        //File reading
        std::string vertex_buffer, fragment_buffer, geometry_buffer;

        readShaderCode(vertexPath, &vertex_buffer);
        const char* vShaderCode = vertex_buffer.c_str();

        readShaderCode(fragmentPath, &fragment_buffer);
        const char* fShaderCode = fragment_buffer.c_str();

        readShaderCode(geometryPath, &geometry_buffer);
        const char* gShaderCode = geometry_buffer.c_str();

        //Compiling
        uint32_t vertex, geometry, fragment;

        vertex = compileShader(GL_VERTEX_SHADER, vShaderCode);
        geometry = compileShader(GL_GEOMETRY_SHADER, gShaderCode);
        fragment = compileShader(GL_FRAGMENT_SHADER, fShaderCode);

        //Linking
        uint32_t shaders[] = { vertex, geometry, fragment };
        linkShader(shaders, 3);

        glDeleteShader(vertex);
        glDeleteShader(fragment);
        glDeleteShader(geometry);

        _has_geometry = true;
        _vertex_path = vertexPath;
        _geometry_path = geometryPath;
        _fragment_path = fragmentPath;
    }

    Shader::~Shader()
    {
        glDeleteProgram(_ID);
    }

    void
    Shader::readShaderCode(const std::string& path, std::string* code)
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
        catch (std::ifstream::failure e)
        {
            std::cerr << "Could not read shader file: " << path;
        }
    }

    uint32_t 
    Shader::compileShader(unsigned int shaderType, const std::string& shader_source)
    {
        uint32_t shader;
        int32_t success;

        shader = glCreateShader(shaderType);
        const char* shader_source_c_str = shader_source.c_str();
        glShaderSource(shader, 1, &shader_source_c_str, NULL);
        glCompileShader(shader);

        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

        if (!success)
        {
            int32_t length;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
            char* infoLog = (char*)malloc(sizeof(char) * length);
            glGetShaderInfoLog(shader, length, &length, infoLog);
            if (shaderType == GL_VERTEX_SHADER)
            {
                std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + std::string(infoLog) + "\n";
            }
            else if (shaderType == GL_FRAGMENT_SHADER)
            {
                std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + std::string(infoLog) + "\n";
            }
            else if (shaderType == GL_GEOMETRY_SHADER)
            {
                std::cerr << "ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n" + std::string(infoLog) + "\n";
            }
            else
            {
                std::cerr << "ERROR::SHADER::COMPILATION_FAILED\nUnknown shader type\n";
            }
            
            free(infoLog);
        }

        return shader;
    }

    void 
    Shader::linkShader(const uint32_t* shaders, const uint32_t& num_shaders)
    {
        int32_t success;

        _ID = glCreateProgram();
        for (uint32_t i = 0; i < num_shaders; ++i)
        {
            glAttachShader(_ID, shaders[i]);
        }
        glLinkProgram(_ID);

        glGetProgramiv(_ID, GL_LINK_STATUS, &success);
        if (!success)
        {
            int32_t length;
            glGetProgramiv(_ID, GL_INFO_LOG_LENGTH, &length);
            char* infoLog = (char*)malloc(sizeof(char) * length);
            glGetProgramInfoLog(_ID, length, &length, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + std::string(infoLog) + "\n";
            free(infoLog);
        }
    }

    void
    Shader::use() const
    {
        glUseProgram(_ID);
    }

    void
    Shader::setInt(const std::string& name, const int& value)
    {
        glUniform1i(glGetUniformLocation(_ID, name.c_str()), value);
    }

    void
    Shader::setFloat(const std::string& name, const float& value)
    {
        glUniform1f(glGetUniformLocation(_ID, name.c_str()), value);
    }

    void
    Shader::setVec2(const std::string& name, const glm::vec2& value)
    {
        glUniform2f(glGetUniformLocation(_ID, name.c_str()), value.x, value.y);
    }

    void
    Shader::setVec3(const std::string& name, const glm::vec3& value)
    {
        glUniform3f(glGetUniformLocation(_ID, name.c_str()), value.x, value.y, value.z);
    }

    void
    Shader::setVec4(const std::string& name, const glm::vec4& value)
    {
        glUniform4f(glGetUniformLocation(_ID, name.c_str()), value.x, value.y, value.z, value.w);
    }

    void
    Shader::setMat4(const std::string&name, const glm::mat4& value)
    {
        glUniformMatrix4fv(glGetUniformLocation(_ID, name.c_str()), 1, GL_FALSE, &value[0][0]);
    }

    void 
    Shader::setMVP(const glm::mat4& M, const glm::mat4& V, const glm::mat4& P)
    {
        setMat4("M", M);
        setMat4("V", V);
        setMat4("P", P);
    }
}