#include <Scene/Components.h>

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
}    // namespace atcg