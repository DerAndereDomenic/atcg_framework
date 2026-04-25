#include <Scene/Components/TransformComponent.h>
#include <Scene/ComponentRegistry.h>

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