#pragma once

#include <Core/glm.h>
#include <Core/UUID.h>
#include <DataStructure/Mesh.h>
#include <DataStructure/PointCloud.h>
#include <Renderer/Shader.h>
#include <Renderer/Camera.h>
#include <Renderer/Renderer.h>

namespace atcg
{
struct TransformComponent
{
    TransformComponent() = default;
    TransformComponent(const glm::vec3& position, const glm::vec3& scale, const glm::vec3& rotation)
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

struct MeshComponent
{
    MeshComponent(const atcg::ref_ptr<Mesh>& mesh) : mesh(mesh) {}

    atcg::ref_ptr<Mesh> mesh;
};

struct PointCloudComponent
{
    PointCloudComponent(const atcg::ref_ptr<PointCloud>& point_cloud) : point_cloud(point_cloud) {}

    atcg::ref_ptr<Mesh> point_cloud;
};

struct GridComponent
{
    GridComponent(const atcg::ref_ptr<VertexBuffer>& points, const atcg::ref_ptr<VertexBuffer>& edges)
        : points(points),
          edges(edges)
    {
    }

    atcg::ref_ptr<VertexBuffer> points;
    atcg::ref_ptr<VertexBuffer> edges;
};

struct RenderComponent
{
    RenderComponent(const atcg::ref_ptr<Shader>& shader,
                    const atcg::ref_ptr<Camera>& camera,
                    const glm::vec3& color,
                    const atcg::DrawMode& draw_mode)
        : shader(shader),
          camera(camera),
          color(color),
          draw_mode(draw_mode)
    {
    }

    atcg::ref_ptr<Shader> shader;
    atcg::ref_ptr<Camera> camera;
    glm::vec3 color;
    atcg::DrawMode draw_mode;
};
}    // namespace atcg