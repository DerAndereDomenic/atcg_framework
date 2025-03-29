#pragma once


#include <Core/glm.h>
#include <Core/UUID.h>
#include <Renderer/Shader.h>
#include <Renderer/Camera.h>
#include <Renderer/Renderer.h>
#include <Renderer/Material.h>
#include <DataStructure/Graph.h>
#include <nanort.h>

#include <vector>
namespace atcg
{
struct TransformComponent
{
    /**
     * @brief Create a transform component
     *
     * @param position The position
     * @param scale The scale
     * @param rotation Rotation in Euler angles
     */
    TransformComponent(const glm::vec3& position = glm::vec3(0),
                       const glm::vec3& scale    = glm::vec3(1),
                       const glm::vec3& rotation = glm::vec3(0))
        : _position(position),
          _scale(scale),
          _rotation(rotation)
    {
        calculateModelMatrix();
    }

    /**
     * @brief Create a transform component
     *
     * @param model The model matrix
     */
    TransformComponent(const glm::mat4& model) : _model_matrix(model) { decomposeModelMatrix(); }

    /**
     * @brief Set the position
     *
     * @param position The position
     */
    ATCG_INLINE void setPosition(const glm::vec3& position)
    {
        _position = position;
        ++_revision;
        calculateModelMatrix();
    }

    /**
     * @brief Set the rotation (Euler angles)
     *
     * @param rotation The euler angles
     */
    ATCG_INLINE void setRotation(const glm::vec3& rotation)
    {
        _rotation = rotation;
        ++_revision;
        calculateModelMatrix();
    }

    /**
     * @brief Set the scale
     *
     * @param scale The scale
     */
    ATCG_INLINE void setScale(const glm::vec3& scale)
    {
        _scale = scale;
        ++_revision;
        calculateModelMatrix();
    }

    /**
     * @brief Set the model matrix
     *
     * @param model The model matrix
     */
    ATCG_INLINE void setModel(const glm::mat4& model)
    {
        _model_matrix = model;
        ++_revision;
        decomposeModelMatrix();
    }

    /**
     * @brief Get the model matrix
     * @return The model matrix
     */
    ATCG_INLINE glm::mat4 getModel() const { return _model_matrix; }

    /**
     * @brief Get the position
     * @return The position
     */
    ATCG_INLINE glm::vec3 getPosition() const { return _position; }

    /**
     * @brief Get the scale
     * @return The scale
     */
    ATCG_INLINE glm::vec3 getScale() const { return _scale; }

    /**
     * @brief Get the rotation
     * @return The rotation
     */
    ATCG_INLINE glm::vec3 getRotation() const { return _rotation; }

    /**
     * @brief Get the number of revision (edits) of the transform component
     * @return The number of revisions
     */
    ATCG_INLINE uint64_t revision() const { return _revision; }

    ATCG_INLINE operator glm::mat4() const { return _model_matrix; }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Transform"; }

private:
    void calculateModelMatrix();
    void decomposeModelMatrix();

private:
    glm::mat4 _model_matrix = glm::mat4(1);
    glm::vec3 _position     = glm::vec3(0);
    glm::vec3 _scale        = glm::vec3(1);
    glm::vec3 _rotation     = glm::vec3(0);    // Euler Angles
    uint64_t _revision      = 0;
};

struct IDComponent
{
    IDComponent() : _ID(UUID()) {}
    IDComponent(uint64_t id) : _ID(UUID(id)) {}

    ATCG_INLINE UUID ID() const { return _ID; }

private:
    UUID _ID;
};

struct NameComponent
{
    NameComponent() = default;
    NameComponent(const std::string& name) : _name(name) {}

    ATCG_INLINE const std::string& name() const { return _name; }

private:
    std::string _name;
};

struct GeometryComponent
{
    GeometryComponent() = default;
    GeometryComponent(const atcg::ref_ptr<Graph>& graph) : graph(graph) {}

    atcg::ref_ptr<Graph> graph;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Geometry"; }
};

struct AccelerationStructureComponent
{
    AccelerationStructureComponent() = default;

    // Don't retrieve this from opengl each time used
    atcg::MemoryBuffer<glm::vec3> vertices;
    atcg::MemoryBuffer<glm::u32vec3> faces;
    nanort::BVHAccel<float> accel;
};

struct CameraComponent
{
    CameraComponent() = default;
    CameraComponent(const atcg::ref_ptr<Camera>& camera, const uint32_t width = 1024, const uint32_t height = 1024)
        : camera(camera),
          width(width),
          height(height)
    {
        if(dynamic_cast<PerspectiveCamera*>(camera.get()))
        {
            perspective = true;
        }
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Camera"; }

    atcg::ref_ptr<Camera> camera;
    glm::vec3 color  = glm::vec3(1);
    bool perspective = false;
    uint32_t width   = 1024;
    uint32_t height  = 1024;
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

struct MeshRenderComponent : public RenderComponent
{
    MeshRenderComponent(const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"))
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_TRIANGLE),
          shader(shader)
    {
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Mesh Renderer"; }

    atcg::ref_ptr<Shader> shader = atcg::ShaderManager::getShader("base");
    Material material;
    bool receive_shadow = true;
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

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Point Renderer"; }

    atcg::ref_ptr<Shader> shader = atcg::ShaderManager::getShader("base");
    glm::vec3 color              = glm::vec3(1);
    float point_size             = 1.0f;
};

struct PointSphereRenderComponent : public RenderComponent
{
    PointSphereRenderComponent(const atcg::ref_ptr<Shader>& shader = atcg::ShaderManager::getShader("base"),
                               const float& point_size             = 0.1f)
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_POINTS_SPHERE),
          shader(shader),
          point_size(point_size)
    {
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Point Sphere Renderer"; }

    atcg::ref_ptr<Shader> shader = atcg::ShaderManager::getShader("base");
    float point_size             = 0.1f;
    Material material;
};

struct EdgeRenderComponent : public RenderComponent
{
    EdgeRenderComponent(const glm::vec3& color = glm::vec3(1))
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_EDGES),
          color(color)
    {
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Edge Renderer"; }

    glm::vec3 color = glm::vec3(1);
};

struct EdgeCylinderRenderComponent : public RenderComponent
{
    EdgeCylinderRenderComponent(float radius = 0.001f)
        : RenderComponent(atcg::DrawMode::ATCG_DRAW_MODE_EDGES_CYLINDER),
          radius(radius)
    {
    }

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Edge Cylinder Renderer"; }

    float radius = 0.001f;
    Material material;
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

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Instance Renderer"; }

    atcg::ref_ptr<VertexBuffer> instance_vbo;
    Material material;
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

struct PointLightComponent
{
    PointLightComponent(const float intensity = 1.0f, const glm::vec3& color = glm::vec3(1))
        : intensity(intensity),
          color(color)
    {
    }

    float intensity  = 1.0f;
    glm::vec3 color  = glm::vec3(1);
    bool cast_shadow = true;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Point Light"; }
};

}    // namespace atcg