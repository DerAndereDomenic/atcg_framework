#pragma once


#include <Core/glm.h>
#include <Core/UUID.h>
#include <Renderer/Shader.h>
#include <Renderer/Camera.h>
#include <Renderer/Renderer.h>
#include <DataStructure/Graph.h>
#include <nanort.h>

#include <vector>
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
    IDComponent(uint64_t id) : ID(UUID(id)) {}

    UUID ID;
};

struct NameComponent
{
    NameComponent() = default;
    NameComponent(const std::string& name) : name(name) {}

    std::string name;
};

struct GeometryComponent
{
    GeometryComponent() = default;
    GeometryComponent(const atcg::ref_ptr<Graph>& graph) : graph(graph) {}

    atcg::ref_ptr<Graph> graph;
};

struct AccelerationStructureComponent
{
    AccelerationStructureComponent() = default;

    // Don't retrieve this from opengl each time used
    atcg::ref_ptr<glm::vec3> vertices;
    atcg::ref_ptr<glm::u32vec3> faces;
    nanort::BVHAccel<float> accel;
};

struct CameraComponent
{
    CameraComponent() = default;
    CameraComponent(const atcg::ref_ptr<Camera>& camera) : camera(camera)
    {
        if(dynamic_cast<PerspectiveCamera*>(camera.get())) { perspective = true; }
    }

    atcg::ref_ptr<Camera> camera;
    bool perspective = false;
};

struct EditorCameraComponent : public CameraComponent
{
    EditorCameraComponent() = default;
    EditorCameraComponent(const atcg::ref_ptr<Camera>& camera) : CameraComponent(camera) {}
};

struct RenderComponent
{
    RenderComponent(atcg::DrawMode draw_mode) : draw_mode(draw_mode) {}

    atcg::DrawMode draw_mode;
    bool visible = true;
};

struct MaterialComponent
{
    MaterialComponent();

    inline atcg::ref_ptr<atcg::Texture2D> getDiffuseTexture() const { return _diffuse_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getNormalTexture() const { return _normal_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getRoughnessTexture() const { return _roughness_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getMetallicTexture() const { return _metallic_texture; }
    inline atcg::ref_ptr<atcg::Texture2D> getDisplacementTexture() const { return _displacement_texture; }

    inline void setDiffuseTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _diffuse_texture = texture; }

    inline void setNormalTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _normal_texture = texture; }

    inline void setRoughnessTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _roughness_texture = texture; }

    inline void setMetallicTexture(const atcg::ref_ptr<atcg::Texture2D>& texture) { _metallic_texture = texture; }

    inline void setDisplacementTexture(const atcg::ref_ptr<atcg::Texture2D>& texture)
    {
        _displacement_texture = texture;
    }

    void setDiffuseColor(const glm::vec4& color);

    void setRoughness(const float roughness);

    void setMetallic(const float metallic);

private:
    atcg::ref_ptr<atcg::Texture2D> _diffuse_texture;
    atcg::ref_ptr<atcg::Texture2D> _normal_texture;
    atcg::ref_ptr<atcg::Texture2D> _roughness_texture;
    atcg::ref_ptr<atcg::Texture2D> _metallic_texture;
    atcg::ref_ptr<atcg::Texture2D> _displacement_texture;
};

struct MeshRenderComponent : public RenderComponent
{
    MeshRenderComponent(const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                        const glm::vec3& color              = glm::vec3(1))
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE),
          shader(shader),
          color(color)
    {
    }

    atcg::ref_ptr<Shader> shader = atcg::ShaderManager::getShader("base");
    glm::vec3 color              = glm::vec3(1);
};

struct PointRenderComponent : public RenderComponent
{
    PointRenderComponent(const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                         const glm::vec3& color              = glm::vec3(1),
                         const float& point_size             = 1.0f)
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_POINTS),
          shader(shader),
          color(color),
          point_size(point_size)
    {
    }

    atcg::ref_ptr<Shader> shader = atcg::ShaderManager::getShader("base");
    glm::vec3 color              = glm::vec3(1);
    float point_size             = 1.0f;
};

struct PointSphereRenderComponent : public RenderComponent
{
    PointSphereRenderComponent(const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                               const glm::vec3& color              = glm::vec3(1),
                               const float& point_size             = 0.1f)
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE),
          shader(shader),
          color(color),
          point_size(point_size)
    {
    }

    atcg::ref_ptr<Shader> shader = atcg::ShaderManager::getShader("base");
    glm::vec3 color              = glm::vec3(1);
    float point_size             = 0.1f;
};

struct EdgeRenderComponent : public RenderComponent
{
    EdgeRenderComponent(const glm::vec3& color = glm::vec3(1))
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_EDGES),
          color(color)
    {
    }

    glm::vec3 color = glm::vec3(1);
};

struct EdgeCylinderRenderComponent : public RenderComponent
{
    EdgeCylinderRenderComponent(const glm::vec3& color = glm::vec3(1), float radius = 1.0f)
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER),
          color(color),
          radius(radius)
    {
    }

    glm::vec3 color = glm::vec3(1);
    float radius    = 1.0f;
};

struct InstanceRenderComponent : public RenderComponent
{
    InstanceRenderComponent() : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_INSTANCED) {}
    InstanceRenderComponent(const std::vector<atcg::Instance>& instances)
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_INSTANCED)
    {
        instance_vbo = atcg::make_ref<VertexBuffer>((void*)instances.data(), instances.size() * sizeof(atcg::Instance));
        instance_vbo->setLayout({{atcg::ShaderDataType::Mat4, "aModel"}, {atcg::ShaderDataType::Float3, "aColor"}});
    }

    atcg::ref_ptr<VertexBuffer> instance_vbo;
};

struct CustomRenderComponent : public RenderComponent
{
    using RenderCallbackFn = std::function<void(Entity, const atcg::ref_ptr<Camera>& camera)>;

    CustomRenderComponent(const RenderCallbackFn& callback, atcg::DrawMode draw_mode)
        : RenderComponent(draw_mode),
          callback(callback)
    {
    }

    RenderCallbackFn callback;
};

}    // namespace atcg