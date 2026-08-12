#pragma once

namespace atcg
{
struct HDRFilmData
{
    uint32_t width;
    uint32_t height;

    glm::vec3* accumulation_buffer;
};
}    // namespace atcg