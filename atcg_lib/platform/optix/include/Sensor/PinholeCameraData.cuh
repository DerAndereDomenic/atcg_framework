#pragma once

#include <Film/FilmVPtrTable.cuh>

namespace atcg
{
struct PinholeCameraData
{
    glm::vec3 cam_eye;
    glm::vec3 U;
    glm::vec3 V;
    glm::vec3 W;
    float fov_y;
    float aspect_ratio;
    glm::vec2 optical_center;
    const FilmVPtrTable* film;
};
}    // namespace atcg