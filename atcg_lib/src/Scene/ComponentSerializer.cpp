#include <Scene/ComponentSerializer.h>

#include <Scene/Entity.h>

#include <DataStructure/TorchUtils.h>

#include <Core/Path.h>

namespace atcg
{

namespace Serialization
{
#define ID_KEY                     "ID"
#define NAME_KEY                   "Name"
#define TRANSFORM_KEY              "Transform"
#define POSITION_KEY               "Position"
#define SCALE_KEY                  "Scale"
#define EULER_ANGLES_KEY           "EulerAngles"
#define SHADER_KEY                 "Shader"
#define MATERIAL_KEY               "Material"
#define GEOMETRY_KEY               "Geometry"
#define VERTICES_KEY               "Vertices"
#define MESH_RENDERER_KEY          "MeshRenderer"
#define POINT_RENDERER_KEY         "PointRenderer"
#define POINT_SPHERE_RENDERER_KEY  "PointSphereRenderer"
#define EDGE_RENDERER_KEY          "EdgeRenderer"
#define EDGE_CYLINDER_RENDERER_KEY "EdgeCylinderRenderer"
#define INSTANCE_RENDERER_KEY      "InstanceRenderer"
#define INSTANCES_KEY              "Instances"
#define COLOR_KEY                  "Color"
#define POINT_SIZE_KEY             "PointSize"
#define RADIUS_KEY                 "Radius"
#define PERSPECTIVE_CAMERA_KEY     "PerspectiveCamera"
#define CAMERA_IMAGE_KEY           "Image"
#define ASPECT_RATIO_KEY           "AspectRatio"
#define FOVY_KEY                   "FoVy"
#define LOOKAT_KEY                 "LookAt"
#define NEAR_KEY                   "Near"
#define FAR_KEY                    "Far"
#define WIDTH_KEY                  "width"
#define HEIGHT_KEY                 "height"
#define PREVIEW_KEY                "preview"
#define OPTICAL_CENTER_KEY         "OpticalCenter"
#define POINT_LIGHT_KEY            "PointLight"
#define INTENSITY_KEY              "Intensity"
#define CAST_SHADOWS_KEY           "CastShadow"
#define RECEIVE_SHADOWS_KEY        "ReceiveShadow"
#define SCRIPT_KEY                 "Script"
#define RENDER_SCALE_KEY           "Scale"
#define LAYOUT_KEY                 "Layout"
#define PATH_KEY                   "Path"
#define EMISSIVE_SCALE_KEY         "EmissiveScale"
#define EMISSIVE_COLOR_KEY         "EmissiveColor"
#define EMISSIVE_TEXTURE_KEY       "EmissiveTexture"
#define MESH_LIGHT_KEY             "MeshLight"

void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size)
{
    std::ofstream summary_file(file_name, std::ios::out | std::ios::binary);
    summary_file.write(data, byte_size);
    summary_file.close();
}

std::vector<uint8_t> deserializeBuffer(const std::string& file_name)
{
    std::ifstream summary_file(file_name, std::ios::in | std::ios::binary);
    std::vector<uint8_t> buffer_char(std::istreambuf_iterator<char>(summary_file), {});
    summary_file.close();

    return buffer_char;
}

nlohmann::json serializeLayout(const atcg::BufferLayout& layout)
{
    nlohmann::json::array_t json_layout;
    for(auto element: layout)
    {
        nlohmann::json::array_t json_element;
        json_element.push_back((int)element.type);
        json_element.push_back(element.name);

        json_layout.push_back(json_element);
    }

    return json_layout;
}

atcg::BufferLayout deserializeLayout(nlohmann::json& layout_node)
{
    std::vector<atcg::BufferElement> elements;
    for(nlohmann::json::array_t element: layout_node)
    {
        atcg::BufferElement buffer_element((atcg::ShaderDataType)element[0], element[1]);
        elements.push_back(buffer_element);
    }

    return atcg::BufferLayout(elements);
}


void ComponentSerializer<IDComponent>::serialize_component(const std::string& file_path,
                                                           const atcg::ref_ptr<Scene>& scene,
                                                           Entity entity,
                                                           IDComponent& component,
                                                           nlohmann::json& j) const
{
    j[ID_KEY] = (uint64_t)entity.getComponent<IDComponent>().ID();
}


void ComponentSerializer<NameComponent>::serialize_component(const std::string& file_path,
                                                             const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             NameComponent& component,
                                                             nlohmann::json& j) const
{
    j[NAME_KEY] = entity.getComponent<NameComponent>().name();
}


void ComponentSerializer<TransformComponent>::serialize_component(const std::string& file_path,
                                                                  const atcg::ref_ptr<Scene>& scene,
                                                                  Entity entity,
                                                                  TransformComponent& component,
                                                                  nlohmann::json& j) const
{
    glm::vec3 position                 = component.getPosition();
    glm::vec3 scale                    = component.getScale();
    glm::vec3 rotation                 = component.getRotation();
    j[TRANSFORM_KEY][POSITION_KEY]     = nlohmann::json::array({position.x, position.y, position.z});
    j[TRANSFORM_KEY][SCALE_KEY]        = nlohmann::json::array({scale.x, scale.y, scale.z});
    j[TRANSFORM_KEY][EULER_ANGLES_KEY] = nlohmann::json::array({rotation.x, rotation.y, rotation.z});
}


void ComponentSerializer<CameraComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               CameraComponent& component,
                                                               nlohmann::json& j) const
{
    atcg::ref_ptr<PerspectiveCamera> cam = std::dynamic_pointer_cast<PerspectiveCamera>(component.camera);

    glm::vec3 position = cam->getPosition();
    glm::vec3 look_at  = cam->getLookAt();
    glm::vec2 offset   = cam->getIntrinsics().opticalCenter();
    float n            = cam->getNear();
    float f            = cam->getFar();

    j[PERSPECTIVE_CAMERA_KEY][ASPECT_RATIO_KEY]   = cam->getAspectRatio();
    j[PERSPECTIVE_CAMERA_KEY][FOVY_KEY]           = cam->getFOV();
    j[PERSPECTIVE_CAMERA_KEY][POSITION_KEY]       = nlohmann::json::array({position.x, position.y, position.z});
    j[PERSPECTIVE_CAMERA_KEY][LOOKAT_KEY]         = nlohmann::json::array({look_at.x, look_at.y, look_at.z});
    j[PERSPECTIVE_CAMERA_KEY][NEAR_KEY]           = n;
    j[PERSPECTIVE_CAMERA_KEY][FAR_KEY]            = f;
    j[PERSPECTIVE_CAMERA_KEY][WIDTH_KEY]          = component.width;
    j[PERSPECTIVE_CAMERA_KEY][HEIGHT_KEY]         = component.height;
    j[PERSPECTIVE_CAMERA_KEY][PREVIEW_KEY]        = component.render_preview;
    j[PERSPECTIVE_CAMERA_KEY][OPTICAL_CENTER_KEY] = nlohmann::json::array({offset.x, offset.y});
    j[PERSPECTIVE_CAMERA_KEY][RENDER_SCALE_KEY]   = component.render_scale;

    if(component.image())
    {
        j[PERSPECTIVE_CAMERA_KEY][CAMERA_IMAGE_KEY] = (uint64_t)component.image_handle;
    }
}


void ComponentSerializer<GeometryComponent>::serialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 GeometryComponent& component,
                                                                 nlohmann::json& j) const
{
    j[GEOMETRY_KEY] = (uint64_t)component.graph_handle;
}


void ComponentSerializer<MeshRenderComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   MeshRenderComponent& component,
                                                                   nlohmann::json& j) const
{
    if(AssetManager::isAssetHandleValid(component.shader_handle))
    {
        j[MESH_RENDERER_KEY][SHADER_KEY] = (uint64_t)component.shader_handle;
    }
    j[MESH_RENDERER_KEY][RECEIVE_SHADOWS_KEY] = component.receive_shadow;

    j[MESH_RENDERER_KEY][MATERIAL_KEY] = (uint64_t)component.material_handle;
}


void ComponentSerializer<PointRenderComponent>::serialize_component(const std::string& file_path,
                                                                    const atcg::ref_ptr<Scene>& scene,
                                                                    Entity entity,
                                                                    PointRenderComponent& component,
                                                                    nlohmann::json& j) const
{
    j[POINT_RENDERER_KEY][COLOR_KEY] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
    j[POINT_RENDERER_KEY][POINT_SIZE_KEY] = component.point_size;
    if(AssetManager::isAssetHandleValid(component.shader_handle))
    {
        j[POINT_RENDERER_KEY][SHADER_KEY] = (uint64_t)component.shader_handle;
    }
}


void ComponentSerializer<PointSphereRenderComponent>::serialize_component(const std::string& file_path,
                                                                          const atcg::ref_ptr<Scene>& scene,
                                                                          Entity entity,
                                                                          PointSphereRenderComponent& component,
                                                                          nlohmann::json& j) const
{
    j[POINT_SPHERE_RENDERER_KEY][POINT_SIZE_KEY] = component.point_size;
    if(AssetManager::isAssetHandleValid(component.shader_handle))
    {
        j[POINT_SPHERE_RENDERER_KEY][SHADER_KEY] = (uint64_t)component.shader_handle;
    }

    j[POINT_SPHERE_RENDERER_KEY][MATERIAL_KEY] = (uint64_t)component.material_handle;
}


void ComponentSerializer<EdgeRenderComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   EdgeRenderComponent& component,
                                                                   nlohmann::json& j) const
{
    j[EDGE_RENDERER_KEY][COLOR_KEY] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
}


void ComponentSerializer<EdgeCylinderRenderComponent>::serialize_component(const std::string& file_path,
                                                                           const atcg::ref_ptr<Scene>& scene,
                                                                           Entity entity,
                                                                           EdgeCylinderRenderComponent& component,
                                                                           nlohmann::json& j) const
{
    j[EDGE_CYLINDER_RENDERER_KEY][RADIUS_KEY] = component.radius;

    j[EDGE_CYLINDER_RENDERER_KEY][MATERIAL_KEY] = (uint64_t)component.material_handle;
}


void ComponentSerializer<InstanceRenderComponent>::serialize_component(const std::string& file_path,
                                                                       const atcg::ref_ptr<Scene>& scene,
                                                                       Entity entity,
                                                                       InstanceRenderComponent& component,
                                                                       nlohmann::json& j) const
{
    auto shader = component.shader() ? component.shader() : atcg::ShaderManager::getShader("instanced");
    if(AssetManager::isAssetHandleValid(component.shader_handle))
    {
        j[INSTANCE_RENDERER_KEY][SHADER_KEY] = (uint64_t)component.shader_handle;
    }
    j[INSTANCE_RENDERER_KEY][MATERIAL_KEY] = (uint64_t)component.material_handle;

    IDComponent& id = entity.getComponent<IDComponent>();
    nlohmann::json::array_t buffers;
    for(int i = 0; i < component.instance_vbos.size(); ++i)
    {
        const char* buffer      = component.instance_vbos[i]->getHostPointer<char>();
        std::string buffer_name = file_path + "." + std::to_string(id.ID()) + ".instance_" + std::to_string(i);
        serializeBuffer(buffer_name, buffer, component.instance_vbos[i]->size());
        nlohmann::json json_buffer;
        json_buffer[PATH_KEY]   = buffer_name;
        json_buffer[LAYOUT_KEY] = serializeLayout(component.instance_vbos[i]->getLayout());
        buffers.push_back(json_buffer);
        component.instance_vbos[i]->unmapHostPointers();
    }

    j[INSTANCE_RENDERER_KEY][INSTANCES_KEY] = buffers;
}


void ComponentSerializer<PointLightComponent>::serialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   PointLightComponent& component,
                                                                   nlohmann::json& j) const
{
    j[POINT_LIGHT_KEY][INTENSITY_KEY] = component.intensity;
    j[POINT_LIGHT_KEY][COLOR_KEY] = nlohmann::json::array({component.color.x, component.color.y, component.color.z});
    j[POINT_LIGHT_KEY][CAST_SHADOWS_KEY] = component.cast_shadow;
}

void ComponentSerializer<MeshLightComponent>::serialize_component(const std::string& file_path,
                                                                  const atcg::ref_ptr<Scene>& scene,
                                                                  Entity entity,
                                                                  MeshLightComponent& component,
                                                                  nlohmann::json& j) const
{
    j[MESH_LIGHT_KEY][EMISSIVE_SCALE_KEY] = component.intensity;

    if(AssetManager::isAssetHandleValid(component.emissive_handle))
    {
        j[MESH_LIGHT_KEY][EMISSIVE_TEXTURE_KEY] = (uint64_t)component.emissive_handle;
    }
    else
    {
        auto data         = component.getEmissiveTexture()->getData(atcg::CPU);
        glm::u8vec3 color = {data.index({0, 0, 0}).item<uint8_t>(),
                             data.index({0, 0, 1}).item<uint8_t>(),
                             data.index({0, 0, 2}).item<uint8_t>()};

        glm::vec3 c(color);
        c = c / 255.0f;

        j[MESH_LIGHT_KEY][EMISSIVE_COLOR_KEY] = nlohmann::json::array({c.x, c.y, c.z});
    }
}


void ComponentSerializer<ScriptComponent>::serialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               ScriptComponent& component,
                                                               nlohmann::json& j) const
{
    j[SCRIPT_KEY] = (uint64_t)component.script_handle;
}


void ComponentSerializer<IDComponent>::deserialize_component(const std::string& file_path,
                                                             const atcg::ref_ptr<Scene>& scene,
                                                             Entity entity,
                                                             nlohmann::json& j) const
{
    if(!j.contains(ID_KEY))
    {
        return;
    }

    entity.addOrReplaceComponent<IDComponent>((uint64_t)j[ID_KEY]);
}


void ComponentSerializer<NameComponent>::deserialize_component(const std::string& file_path,
                                                               const atcg::ref_ptr<Scene>& scene,
                                                               Entity entity,
                                                               nlohmann::json& j) const
{
    if(!j.contains(NAME_KEY))
    {
        return;
    }

    entity.addOrReplaceComponent<NameComponent>(j[NAME_KEY]);
}


void ComponentSerializer<TransformComponent>::deserialize_component(const std::string& file_path,
                                                                    const atcg::ref_ptr<Scene>& scene,
                                                                    Entity entity,
                                                                    nlohmann::json& j) const
{
    if(!j.contains(TRANSFORM_KEY))
    {
        return;
    }

    std::vector<float> position = j[TRANSFORM_KEY].value(POSITION_KEY, std::vector<float> {0.0f, 0.0f, 0.0f});
    std::vector<float> scale    = j[TRANSFORM_KEY].value(SCALE_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    std::vector<float> rotation = j[TRANSFORM_KEY].value(EULER_ANGLES_KEY, std::vector<float> {0.0f, 0.0f, 0.0f});

    entity.addComponent<atcg::TransformComponent>(glm::make_vec3(position.data()),
                                                  glm::make_vec3(scale.data()),
                                                  glm::make_vec3(rotation.data()));
}


void ComponentSerializer<CameraComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(!j.contains(PERSPECTIVE_CAMERA_KEY))
    {
        return;
    }

    float aspect_ratio          = j[PERSPECTIVE_CAMERA_KEY].value(ASPECT_RATIO_KEY, 1.0f);
    float fov                   = j[PERSPECTIVE_CAMERA_KEY].value(FOVY_KEY, 60.0f);
    float n                     = j[PERSPECTIVE_CAMERA_KEY].value(NEAR_KEY, 0.01f);
    float f                     = j[PERSPECTIVE_CAMERA_KEY].value(FAR_KEY, 1000.0f);
    std::vector<float> position = j[PERSPECTIVE_CAMERA_KEY].value(POSITION_KEY, std::vector<float> {0.0f, 0.0f, -1.0f});
    std::vector<float> lookat   = j[PERSPECTIVE_CAMERA_KEY].value(LOOKAT_KEY, std::vector<float> {0.0f, 0.0f, 0.0f});
    std::vector<float> offset   = j[PERSPECTIVE_CAMERA_KEY].value(OPTICAL_CENTER_KEY, std::vector<float> {0.0f, 0.0f});

    CameraExtrinsics extrinsics(glm::make_vec3(position.data()), glm::make_vec3(lookat.data()));
    CameraIntrinsics intrinsics(aspect_ratio, fov, n, f);
    intrinsics.setOpticalCenter(glm::make_vec2(offset.data()));

    auto cam = atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics);

    auto& component          = entity.addComponent<CameraComponent>(cam);
    component.width          = j[PERSPECTIVE_CAMERA_KEY].value(WIDTH_KEY, 1024);
    component.height         = j[PERSPECTIVE_CAMERA_KEY].value(HEIGHT_KEY, 1024);
    component.render_preview = j[PERSPECTIVE_CAMERA_KEY].value(PREVIEW_KEY, false);
    component.render_scale   = j[PERSPECTIVE_CAMERA_KEY].value(RENDER_SCALE_KEY, 1.0f);


    if(j[PERSPECTIVE_CAMERA_KEY].contains(CAMERA_IMAGE_KEY))
    {
        component.image_handle = (AssetHandle)j[PERSPECTIVE_CAMERA_KEY][CAMERA_IMAGE_KEY];
    }
}


void ComponentSerializer<GeometryComponent>::deserialize_component(const std::string& file_path,
                                                                   const atcg::ref_ptr<Scene>& scene,
                                                                   Entity entity,
                                                                   nlohmann::json& j) const
{
    if(!j.contains(GEOMETRY_KEY))
    {
        return;
    }

    auto& geometry        = entity.addComponent<GeometryComponent>();
    geometry.graph_handle = (AssetHandle)j[GEOMETRY_KEY];
}


void ComponentSerializer<MeshRenderComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(MESH_RENDERER_KEY))
    {
        return;
    }

    auto& renderer        = j[MESH_RENDERER_KEY];
    auto& renderComponent = entity.addComponent<MeshRenderComponent>();

    if(j[MESH_RENDERER_KEY].contains(SHADER_KEY))
    {
        renderComponent.shader_handle = (AssetHandle)j[MESH_RENDERER_KEY][SHADER_KEY];
    }


    if(renderer.contains(MATERIAL_KEY))
    {
        renderComponent.material_handle = (AssetHandle)renderer[MATERIAL_KEY];
    }

    renderComponent.receive_shadow = renderer.value(RECEIVE_SHADOWS_KEY, true);
}


void ComponentSerializer<PointRenderComponent>::deserialize_component(const std::string& file_path,
                                                                      const atcg::ref_ptr<Scene>& scene,
                                                                      Entity entity,
                                                                      nlohmann::json& j) const
{
    if(!j.contains(POINT_RENDERER_KEY))
    {
        return;
    }

    auto& renderer             = j[POINT_RENDERER_KEY];
    auto& renderComponent      = entity.addComponent<PointRenderComponent>();
    std::vector<float> color   = renderer.value(COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    renderComponent.color      = glm::make_vec3(color.data());
    renderComponent.point_size = renderer.value(POINT_SIZE_KEY, 1.0f);

    if(j[POINT_RENDERER_KEY].contains(SHADER_KEY))
    {
        renderComponent.shader_handle = (AssetHandle)j[POINT_RENDERER_KEY][SHADER_KEY];
    }
}


void ComponentSerializer<PointSphereRenderComponent>::deserialize_component(const std::string& file_path,
                                                                            const atcg::ref_ptr<Scene>& scene,
                                                                            Entity entity,
                                                                            nlohmann::json& j) const
{
    if(!j.contains(POINT_SPHERE_RENDERER_KEY))
    {
        return;
    }

    auto& renderer             = j[POINT_SPHERE_RENDERER_KEY];
    auto& renderComponent      = entity.addComponent<PointSphereRenderComponent>();
    renderComponent.point_size = renderer.value(POINT_SIZE_KEY, 0.1f);

    if(j[POINT_SPHERE_RENDERER_KEY].contains(SHADER_KEY))
    {
        renderComponent.shader_handle = (AssetHandle)j[POINT_SPHERE_RENDERER_KEY][SHADER_KEY];
    }


    if(renderer.contains(MATERIAL_KEY))
    {
        renderComponent.material_handle = (AssetHandle)renderer[MATERIAL_KEY];
    }
}


void ComponentSerializer<EdgeRenderComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(EDGE_RENDERER_KEY))
    {
        return;
    }

    auto& renderer           = j[EDGE_RENDERER_KEY];
    auto& renderComponent    = entity.addComponent<EdgeRenderComponent>();
    std::vector<float> color = renderer.value(COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    renderComponent.color    = glm::make_vec3(color.data());
}


void ComponentSerializer<EdgeCylinderRenderComponent>::deserialize_component(const std::string& file_path,
                                                                             const atcg::ref_ptr<Scene>& scene,
                                                                             Entity entity,
                                                                             nlohmann::json& j) const
{
    if(!j.contains(EDGE_CYLINDER_RENDERER_KEY))
    {
        return;
    }

    auto& renderer         = j[EDGE_CYLINDER_RENDERER_KEY];
    auto& renderComponent  = entity.addComponent<EdgeCylinderRenderComponent>();
    renderComponent.radius = renderer.value(RADIUS_KEY, 0.001f);


    if(renderer.contains(MATERIAL_KEY))
    {
        renderComponent.material_handle = (AssetHandle)renderer[MATERIAL_KEY];
    }
}


void ComponentSerializer<InstanceRenderComponent>::deserialize_component(const std::string& file_path,
                                                                         const atcg::ref_ptr<Scene>& scene,
                                                                         Entity entity,
                                                                         nlohmann::json& j) const
{
    if(!j.contains(INSTANCE_RENDERER_KEY))
    {
        return;
    }

    auto& renderer        = j[INSTANCE_RENDERER_KEY];
    auto& renderComponent = entity.addComponent<InstanceRenderComponent>();
    if(j[INSTANCE_RENDERER_KEY].contains(SHADER_KEY))
    {
        renderComponent.shader_handle = (AssetHandle)j[INSTANCE_RENDERER_KEY][SHADER_KEY];
    }


    if(renderer.contains(MATERIAL_KEY))
    {
        renderComponent.material_handle = (AssetHandle)renderer[MATERIAL_KEY];
    }

    if(renderer.contains(INSTANCES_KEY))
    {
        nlohmann::json::array_t instances = renderer[INSTANCES_KEY];
        renderComponent.instance_vbos.reserve(instances.size());

        for(auto instance: instances)
        {
            std::string path                      = instance[PATH_KEY];
            atcg::BufferLayout layout             = deserializeLayout(instance[LAYOUT_KEY]);
            std::vector<uint8_t> buffer           = deserializeBuffer(path);
            atcg::ref_ptr<atcg::VertexBuffer> vbo = atcg::make_ref<atcg::VertexBuffer>(buffer.data(), buffer.size());
            vbo->setLayout(layout);
            renderComponent.addInstanceBuffer(vbo);
        }
    }
}


void ComponentSerializer<PointLightComponent>::deserialize_component(const std::string& file_path,
                                                                     const atcg::ref_ptr<Scene>& scene,
                                                                     Entity entity,
                                                                     nlohmann::json& j) const
{
    if(!j.contains(POINT_LIGHT_KEY))
    {
        return;
    }

    auto& point_light           = j[POINT_LIGHT_KEY];
    auto& renderComponent       = entity.addComponent<PointLightComponent>();
    renderComponent.intensity   = point_light.value(INTENSITY_KEY, 1.0f);
    auto color                  = point_light.value(COLOR_KEY, std::vector<float> {1.0f, 1.0f, 1.0f});
    renderComponent.color       = glm::make_vec3(color.data());
    renderComponent.cast_shadow = point_light.value(CAST_SHADOWS_KEY, true);
}

void ComponentSerializer<MeshLightComponent>::deserialize_component(const std::string& file_path,
                                                                    const atcg::ref_ptr<Scene>& scene,
                                                                    Entity entity,
                                                                    nlohmann::json& j) const
{
    if(!j.contains(MESH_LIGHT_KEY))
    {
        return;
    }

    auto& renderComponent = entity.addComponent<MeshLightComponent>();

    renderComponent.intensity = j[MESH_LIGHT_KEY].value(EMISSIVE_SCALE_KEY, 1.0f);
    if(j[MESH_LIGHT_KEY].contains(EMISSIVE_COLOR_KEY))
    {
        std::vector<float> emissive_color = j[MESH_LIGHT_KEY][EMISSIVE_COLOR_KEY];
        renderComponent.setEmissiveColor(glm::make_vec3(emissive_color.data()));
    }
    else if(j[MESH_LIGHT_KEY].contains(EMISSIVE_TEXTURE_KEY))
    {
        AssetHandle emissive_handle     = (AssetHandle)j[MESH_LIGHT_KEY][EMISSIVE_TEXTURE_KEY];
        renderComponent.emissive_handle = emissive_handle;
    }
}


void ComponentSerializer<ScriptComponent>::deserialize_component(const std::string& file_path,
                                                                 const atcg::ref_ptr<Scene>& scene,
                                                                 Entity entity,
                                                                 nlohmann::json& j) const
{
    if(!j.contains(SCRIPT_KEY))
    {
        return;
    }

    auto& script = entity.addComponent<ScriptComponent>();

    script.script_handle = (AssetHandle)j[SCRIPT_KEY];

    script.script()->onAttach(scene, entity);
}

void ComponentSerializer<HomogeneousMediumComponent>::serialize_component(const std::string& file_path,
                                                                          const atcg::ref_ptr<Scene>& scene,
                                                                          Entity entity,
                                                                          HomogeneousMediumComponent& component,
                                                                          nlohmann::json& j) const
{
    glm::vec3 albedo   = component.albedo;
    float density      = component.density;
    float g            = component.g;
    float Le           = component.Le;
    glm::vec3 Le_color = component.Le_color;

    j["Homogeneous Medium"]["albedo"]   = nlohmann::json::array({albedo.x, albedo.y, albedo.z});
    j["Homogeneous Medium"]["density"]  = density;
    j["Homogeneous Medium"]["g"]        = g;
    j["Homogeneous Medium"]["Le"]       = Le;
    j["Homogeneous Medium"]["Le_color"] = nlohmann::json::array({Le_color.x, Le_color.y, Le_color.z});
}


void ComponentSerializer<HomogeneousMediumComponent>::deserialize_component(const std::string& file_path,
                                                                            const atcg::ref_ptr<Scene>& scene,
                                                                            Entity entity,
                                                                            nlohmann::json& j) const
{
    if(!j.contains("Homogeneous Medium"))
    {
        return;
    }

    std::vector<float> albedo   = j["Homogeneous Medium"].value("albedo", std::vector<float> {1.0f, 1.0f, 1.0f});
    std::vector<float> Le_color = j["Homogeneous Medium"].value("Le_color", std::vector<float> {1.0f, 1.0f, 1.0f});

    auto& component    = entity.addComponent<HomogeneousMediumComponent>();
    component.albedo   = glm::make_vec3(albedo.data());
    component.g        = j["Homogeneous Medium"]["g"];
    component.Le       = j["Homogeneous Medium"]["Le"];
    component.Le_color = glm::make_vec3(Le_color.data());
    component.density  = j["Homogeneous Medium"]["density"];
}

void ComponentSerializer<HeterogeneousMediumComponent>::serialize_component(const std::string& file_path,
                                                                            const atcg::ref_ptr<Scene>& scene,
                                                                            Entity entity,
                                                                            HeterogeneousMediumComponent& component,
                                                                            nlohmann::json& j) const
{
    nlohmann::json j_density_grid;
    nlohmann::json j_albedo_grid;
    nlohmann::json j_emission_grid;

    j_density_grid["grid"]  = (uint64_t)component.density_grid.handle;
    j_density_grid["scale"] = component.density_grid.scale;
    nlohmann::json j_bbox_density;
    j_bbox_density["min"] = nlohmann::json::array(
        {component.density_grid.bbox.min.x, component.density_grid.bbox.min.y, component.density_grid.bbox.min.z});
    j_bbox_density["max"] = nlohmann::json::array(
        {component.density_grid.bbox.max.x, component.density_grid.bbox.max.y, component.density_grid.bbox.max.z});
    j_density_grid["bbox"] = j_bbox_density;

    j_albedo_grid["grid"]  = (uint64_t)component.albedo_grid.handle;
    j_albedo_grid["scale"] = component.albedo_grid.scale;
    nlohmann::json j_bbox_albedo;
    j_bbox_albedo["min"] = nlohmann::json::array(
        {component.albedo_grid.bbox.min.x, component.albedo_grid.bbox.min.y, component.albedo_grid.bbox.min.z});
    j_bbox_albedo["max"] = nlohmann::json::array(
        {component.albedo_grid.bbox.max.x, component.albedo_grid.bbox.max.y, component.albedo_grid.bbox.max.z});
    j_albedo_grid["bbox"] = j_bbox_albedo;

    j_emission_grid["grid"]  = (uint64_t)component.emission_grid.handle;
    j_emission_grid["scale"] = component.emission_grid.scale;
    nlohmann::json j_bbox_emission;
    j_bbox_emission["min"] = nlohmann::json::array(
        {component.emission_grid.bbox.min.x, component.emission_grid.bbox.min.y, component.emission_grid.bbox.min.z});
    j_bbox_emission["max"] = nlohmann::json::array(
        {component.emission_grid.bbox.max.x, component.emission_grid.bbox.max.y, component.emission_grid.bbox.max.z});
    j_emission_grid["bbox"] = j_bbox_emission;

    j["Heterogeneous Medium"]["density_grid"]  = j_density_grid;
    j["Heterogeneous Medium"]["albedo_grid"]   = j_albedo_grid;
    j["Heterogeneous Medium"]["emission_grid"] = j_emission_grid;
    j["Heterogeneous Medium"]["g"]             = component.g;
}


void ComponentSerializer<HeterogeneousMediumComponent>::deserialize_component(const std::string& file_path,
                                                                              const atcg::ref_ptr<Scene>& scene,
                                                                              Entity entity,
                                                                              nlohmann::json& j) const
{
    if(!j.contains("Heterogeneous Medium"))
    {
        return;
    }

    auto& component = entity.addComponent<HeterogeneousMediumComponent>();

    // --- Density grid ---
    const auto& j_density         = j["Heterogeneous Medium"]["density_grid"];
    component.density_grid.handle = (AssetHandle)j_density["grid"].get<uint64_t>();
    component.density_grid.scale  = j_density["scale"].get<float>();

    const auto& j_bbox_density      = j_density["bbox"];
    component.density_grid.bbox.min = glm::vec3(j_bbox_density["min"][0].get<float>(),
                                                j_bbox_density["min"][1].get<float>(),
                                                j_bbox_density["min"][2].get<float>());
    component.density_grid.bbox.max = glm::vec3(j_bbox_density["max"][0].get<float>(),
                                                j_bbox_density["max"][1].get<float>(),
                                                j_bbox_density["max"][2].get<float>());

    // --- Albedo grid ---
    const auto& j_albedo         = j["Heterogeneous Medium"]["albedo_grid"];
    component.albedo_grid.handle = (AssetHandle)j_albedo["grid"].get<uint64_t>();
    component.albedo_grid.scale  = j_albedo["scale"].get<float>();

    const auto& j_bbox_albedo      = j_albedo["bbox"];
    component.albedo_grid.bbox.min = glm::vec3(j_bbox_albedo["min"][0].get<float>(),
                                               j_bbox_albedo["min"][1].get<float>(),
                                               j_bbox_albedo["min"][2].get<float>());
    component.albedo_grid.bbox.max = glm::vec3(j_bbox_albedo["max"][0].get<float>(),
                                               j_bbox_albedo["max"][1].get<float>(),
                                               j_bbox_albedo["max"][2].get<float>());

    // --- Emission grid ---
    const auto& j_emission         = j["Heterogeneous Medium"]["emission_grid"];
    component.emission_grid.handle = (AssetHandle)j_emission["grid"].get<uint64_t>();
    component.emission_grid.scale  = j_emission["scale"].get<float>();

    const auto& j_bbox_emission      = j_emission["bbox"];
    component.emission_grid.bbox.min = glm::vec3(j_bbox_emission["min"][0].get<float>(),
                                                 j_bbox_emission["min"][1].get<float>(),
                                                 j_bbox_emission["min"][2].get<float>());
    component.emission_grid.bbox.max = glm::vec3(j_bbox_emission["max"][0].get<float>(),
                                                 j_bbox_emission["max"][1].get<float>(),
                                                 j_bbox_emission["max"][2].get<float>());

    component.g = j["Heterogeneous Medium"]["g"].get<float>();
}
}    // namespace Serialization
}    // namespace atcg