#pragma once

#include <Film/FilmVPtrTable.cuh>

namespace atcg
{
struct HemisphereCameraData
{
    glm::vec3 cam_eye;
    glm::vec3 normal;
    float exposure;

    const FilmVPtrTable* film;
};
}    // namespace atcg