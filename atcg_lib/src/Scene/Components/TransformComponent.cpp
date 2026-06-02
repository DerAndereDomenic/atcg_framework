#include <Scene/Components/TransformComponent.h>
#include <Scene/ComponentRegistry.h>

#define TRANSFORM_KEY    "Transform"
#define POSITION_KEY     "Position"
#define SCALE_KEY        "Scale"
#define EULER_ANGLES_KEY "EulerAngles"

namespace atcg
{
void TransformComponent::calculateModelMatrix()
{
    glm::mat4 scale     = glm::scale(_scale);
    glm::mat4 translate = glm::translate(_position);
    glm::mat4 rotation  = glm::eulerAngleXYZ(_rotation.x, _rotation.y, _rotation.z);

    _model_matrix = translate * rotation * scale;
}

void TransformComponent::decomposeModelMatrix()
{
    _position     = _model_matrix[3];
    glm::mat4 RS  = glm::mat3(_model_matrix);
    float scale_x = glm::length(RS[0]);
    float scale_y = glm::length(RS[1]);
    float scale_z = glm::length(RS[2]);
    _scale        = glm::vec3(scale_x, scale_y, scale_z);
    glm::extractEulerAngleXYZ(glm::mat4(RS * glm::scale(1.0f / _scale)), _rotation.x, _rotation.y, _rotation.z);
}

namespace Serialization
{
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

}    // namespace Serialization

namespace GUI
{
bool displayTransform(const std::string& id, TransformComponent& transform)
{
#ifndef ATCG_HEADLESS
    bool updated       = false;
    glm::vec3 position = transform.getPosition();
    std::stringstream label;
    label << "Position##" << id;
    if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(position), 0.05f))
    {
        transform.setPosition(position);
        updated = true;
    }
    glm::vec3 scale = transform.getScale();
    label.str(std::string());
    label << "Scale##" << id;
    if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(scale), 0.05f, 1e-5f, FLT_MAX))
    {
        scale = glm::clamp(scale, 1e-5f, FLT_MAX);
        transform.setScale(scale);
        updated = true;
    }
    glm::vec3 rotation = glm::degrees(transform.getRotation());
    label.str(std::string());
    label << "Rotation##" << id;
    if(ImGui::DragFloat3(label.str().c_str(), glm::value_ptr(rotation), 0.05f))
    {
        transform.setRotation(glm::radians(rotation));
        updated = true;
    }

    return updated;
#else
    return false;
#endif
}

void ComponentGUIRenderer<TransformComponent>::draw_component(const atcg::ref_ptr<Scene>& scene,
                                                              Entity entity,
                                                              TransformComponent& transform) const
{
#ifndef ATCG_HEADLESS
    std::string id = std::to_string(entity.getComponent<IDComponent>().ID());

    TransformComponent transform_ = transform;

    bool updated = displayTransform(id, transform_);

    if(updated)
    {
        RevisionStack::startRecording<ComponentEditedRevision<TransformComponent>>(scene, entity);
        transform = transform_;
        atcg::RevisionStack::endRecording();
    }
#endif
}
}    // namespace GUI

ATCG_REGISTER_COMPONENT(TransformComponent);
}    // namespace atcg