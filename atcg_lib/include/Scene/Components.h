#pragma once

#include <vector>

#include <Core/glm.h>
#include <Core/UUID.h>
#include <Renderer/Shader.h>
#include <Renderer/Camera.h>
#include <Renderer/Renderer.h>
#include <DataStructure/Graph.h>

namespace atcg
{
struct TransformComponent
{
    TransformComponent(const glm::vec3& position = glm::vec3(0),
                       const glm::vec3& scale    = glm::vec3(1),
                       const glm::vec3& rotation = glm::vec3(0))
        : _position(position),
          _scale(scale),
          _rotation(rotation)
    {
        calculateModelMatrix();
    }

    TransformComponent(const glm::mat4& model) : _model_matrix(model) { decomposeModelMatrix(); }


    inline void setPosition(const glm::vec3& position)
    {
        _position = position;
        calculateModelMatrix();
    }

    inline void setRotation(const glm::vec3& rotation)
    {
        _rotation = rotation;
        calculateModelMatrix();
    }

    inline void setScale(const glm::vec3& scale)
    {
        _scale = scale;
        calculateModelMatrix();
    }

    inline void setModel(const glm::mat4& model)
    {
        _model_matrix = model;
        decomposeModelMatrix();
    }

    inline glm::mat4 getModel() const { return _model_matrix; }

    inline glm::vec3 getPosition() const { return _position; }

    inline glm::vec3 getScale() const { return _scale; }

    inline glm::vec3 getRotation() const { return _rotation; }

    inline operator glm::mat4() const { return _model_matrix; }

private:
    void calculateModelMatrix();
    void decomposeModelMatrix();

private:
    glm::mat4 _model_matrix = glm::mat4(1);
    glm::vec3 _position     = glm::vec3(0);
    glm::vec3 _scale        = glm::vec3(1);
    glm::vec3 _rotation     = glm::vec3(0);    // Euler Angles
};

struct IDComponent
{
    IDComponent() : ID(UUID()) {}

    UUID ID;
};

struct RenderConfig
{
    RenderConfig(const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                 const glm::vec3& color              = glm::vec3(1),
                 const atcg::DrawMode& draw_mode     = atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE)
        : shader(shader),
          color(color),
          draw_mode(draw_mode)
    {
    }

    atcg::ref_ptr<Shader> shader;
    glm::vec3 color;
    atcg::DrawMode draw_mode;
};

struct GeometryComponent
{
    GeometryComponent() = default;
    GeometryComponent(const atcg::ref_ptr<Graph>& graph) : graph(graph) {}

    inline GeometryComponent& addConfig(const RenderConfig& config = {})
    {
        configs.push_back(config);
        return *this;
    }

    atcg::ref_ptr<Graph> graph;
    std::vector<RenderConfig> configs;
};
}    // namespace atcg