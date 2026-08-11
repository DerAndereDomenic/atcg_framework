#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>

namespace atcg
{

struct BSDFVPtrTable;
struct EmitterVPtrTable;
struct MediumVPtrTable;

struct Interaction
{
    glm::vec3 position;
    glm::vec3 incoming_direction;
    float incoming_distance;
    float pdf;

    ATCG_HOST_DEVICE Interaction()
        : position(glm::vec3(std::numeric_limits<float>::signaling_NaN())),
          incoming_direction(glm::vec3(std::numeric_limits<float>::signaling_NaN())),
          incoming_distance(std::numeric_limits<float>::signaling_NaN()),
          pdf(0.0f)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE bool isValid() const { return !glm::isnan(incoming_distance); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInvalid() { incoming_distance = std::numeric_limits<float>::signaling_NaN(); }

    ATCG_INLINE ATCG_HOST_DEVICE bool isFinite() const { return glm::isfinite(incoming_distance); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInfinite() { incoming_distance = std::numeric_limits<float>::infinity(); }
};

struct SurfaceInteraction : public Interaction
{
    ATCG_HOST_DEVICE SurfaceInteraction() : Interaction() {}

    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 barys;
    glm::vec2 uv;
    uint32_t primitive_idx;
    uint32_t entity_id;

    const BSDFVPtrTable *bsdf;
    const EmitterVPtrTable *emitter;
    const MediumVPtrTable *inside_medium;
    const MediumVPtrTable *outside_medium;
};

struct MediumInteraction : public Interaction
{
    ATCG_HOST_DEVICE MediumInteraction() : Interaction() {}
};

struct AnyInteraction
{
    enum Type
    {
        InteractionType,
        SurfaceInteractionType,
        MediumInteractionType
    };
    Type type;

    union
    {
        Interaction it;
        SurfaceInteraction si;
        MediumInteraction mi;
    };

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction() : type(InteractionType), it(Interaction()) {}

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction(const Interaction &interaction) : type(InteractionType), it(interaction)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction(const SurfaceInteraction &surface)
        : type(SurfaceInteractionType),
          si(surface)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction(const MediumInteraction &medium)
        : type(MediumInteractionType),
          mi(medium)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction(const AnyInteraction &ai)
    {
        this->type = ai.type;
        switch(ai.type)
        {
            case InteractionType:
            {
                this->it = ai.it;
                break;
            }
            case MediumInteractionType:
            {
                this->mi = ai.mi;
                break;
            }
            case SurfaceInteractionType:
            {
                this->si = ai.si;
                break;
            }
        }
    }

    ATCG_INLINE ATCG_HOST_DEVICE operator Interaction &() { return it; }
    ATCG_INLINE ATCG_HOST_DEVICE operator const Interaction &() const { return it; }
    ATCG_INLINE ATCG_HOST_DEVICE operator SurfaceInteraction &() { return si; }
    ATCG_INLINE ATCG_HOST_DEVICE operator const SurfaceInteraction &() const { return si; }
    ATCG_INLINE ATCG_HOST_DEVICE operator MediumInteraction &() { return mi; }
    ATCG_INLINE ATCG_HOST_DEVICE operator const MediumInteraction &() const { return mi; }

    ATCG_INLINE ATCG_HOST_DEVICE Interaction *operator->() { return &it; }
    ATCG_INLINE ATCG_HOST_DEVICE const Interaction *operator->() const { return &it; }

    ATCG_INLINE ATCG_HOST_DEVICE bool is_surface() const { return type == SurfaceInteractionType; }
    ATCG_INLINE ATCG_HOST_DEVICE bool is_medium() const { return type == MediumInteractionType; }

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction &operator=(const Interaction &it)
    {
        this->type = InteractionType;
        this->it   = it;
        return *this;
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction &operator=(const MediumInteraction &mi)
    {
        this->type = MediumInteractionType;
        this->mi   = mi;
        return *this;
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction &operator=(const SurfaceInteraction &si)
    {
        this->type = SurfaceInteractionType;
        this->si   = si;
        return *this;
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyInteraction &operator=(const AnyInteraction &ai)
    {
        this->type = ai.type;
        switch(ai.type)
        {
            case InteractionType:
            {
                this->it = ai.it;
                break;
            }
            case MediumInteractionType:
            {
                this->mi = ai.mi;
                break;
            }
            case SurfaceInteractionType:
            {
                this->si = ai.si;
                break;
            }
        }
        return *this;
    }
};

}    // namespace atcg