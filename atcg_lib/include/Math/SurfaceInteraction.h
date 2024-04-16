#pragma once

namespace atcg
{

struct BSDFVPtrTable;
struct EmitterVPtrTable;

struct SurfaceInteraction
{
    bool valid = false;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 barys;
    glm::vec2 uv;
    glm::vec3 incoming_direction;
    float incoming_distance;
    uint32_t primitive_idx;

    const BSDFVPtrTable* bsdf;
    const EmitterVPtrTable* emitter;
};
}    // namespace atcg