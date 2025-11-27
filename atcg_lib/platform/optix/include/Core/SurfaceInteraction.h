#pragma once

#include <Core/glm.h>
#include <CuDiff/CuDiff.h>
#include <Core/Platform.h>
#include <Core/CUDA.h>

namespace atcg
{

struct BSDFVPtrTable;
struct EmitterVPtrTable;
struct MediumVPtrTable;

struct Interaction
{
    bool valid = false;
    glm::vec3 position;
    glm::vec3 incoming_direction;
    float incoming_distance;
};

struct SurfaceInteraction : public Interaction
{
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 barys;
    glm::vec2 uv;
    uint32_t primitive_idx;
    uint32_t entity_id;

    const BSDFVPtrTable* bsdf;
    const EmitterVPtrTable* emitter;
    const MediumVPtrTable* inside_medium;
    const MediumVPtrTable* outside_medium;
};

struct MediumInteraction : public Interaction
{
};

struct DualSurfaceInteraction
{
    bool valid = false;
    // Output
    CuDiff::Dual<4, glm::vec3> position;
    CuDiff::Dual<4, glm::vec3> normal;
    CuDiff::Dual<4, glm::vec2> uv;
    CuDiff::Dual<4, float> incoming_distance;
    CuDiff::Dual<4, float> u_surface;
    CuDiff::Dual<4, float> v_surface;

    glm::vec3 P0, P1, P2;

    // Input
    CuDiff::Dual<4, glm::vec3> incoming_direction;
    CuDiff::Dual<4, glm::vec3> incoming_position;
    CuDiff::Dual<4, float> phi, theta;

    uint32_t primitive_idx;
    uint32_t entity_id;

    const BSDFVPtrTable* bsdf;
    const EmitterVPtrTable* emitter;
    const MediumVPtrTable* inside_medium;
    const MediumVPtrTable* outside_medium;

    ATCG_DEVICE ATCG_INLINE SurfaceInteraction toSi() const
    {
        SurfaceInteraction si;
        si.valid              = valid;
        si.position           = position.val();
        si.normal             = normal.val();
        si.uv                 = uv.val();
        si.incoming_direction = incoming_direction.val();
        si.incoming_distance  = incoming_distance.val();
        si.primitive_idx      = primitive_idx;
        si.entity_id          = entity_id;
        si.bsdf               = bsdf;
        si.emitter            = emitter;
        si.inside_medium      = inside_medium;
        si.outside_medium     = outside_medium;
        si.color              = glm::vec3(1);    // TODO

        return si;
    }
};
}    // namespace atcg