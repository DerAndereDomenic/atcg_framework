#include <Scene/Components.h>
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

ATCG_REGISTER_COMPONENT_SERIALIZATION(IDComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(NameComponent);
ATCG_REGISTER_COMPONENT(TransformComponent);
ATCG_REGISTER_COMPONENT(CameraComponent);
ATCG_REGISTER_COMPONENT(GeometryComponent);
ATCG_REGISTER_COMPONENT(MeshRenderComponent);
ATCG_REGISTER_COMPONENT(PointRenderComponent);
ATCG_REGISTER_COMPONENT(PointSphereRenderComponent);
ATCG_REGISTER_COMPONENT(EdgeRenderComponent);
ATCG_REGISTER_COMPONENT(EdgeCylinderRenderComponent);
ATCG_REGISTER_COMPONENT(InstanceRenderComponent);
ATCG_REGISTER_COMPONENT(PointLightComponent);
ATCG_REGISTER_COMPONENT(MeshLightComponent);
ATCG_REGISTER_COMPONENT(ScriptComponent);
ATCG_REGISTER_COMPONENT_DRAW(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HomogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_DRAW(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_STORE(HeterogeneousMediumComponent);
ATCG_REGISTER_COMPONENT_SERIALIZATION(HeterogeneousMediumComponent);
}    // namespace atcg