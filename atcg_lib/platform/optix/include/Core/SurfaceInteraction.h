#pragma once

#include <Core/glm.h>
#include <CuDiff/CuDiff.h>
#include <Core/Platform.h>
#include <Core/CUDA.h>
#include <DataStructure/Frame.h>
#include <Math/mat6.h>

namespace atcg
{

struct BSDFVPtrTable;
struct EmitterVPtrTable;
struct MediumVPtrTable;

struct Interaction
{
    glm::vec3 position;
    glm::vec3 incoming_direction;
    atcg::Frame<glm::vec3> reference_frame;
    float incoming_distance;
    float pdf;

    ATCG_HOST_DEVICE Interaction()
        : position(glm::vec3(std::numeric_limits<float>::signaling_NaN())),
          incoming_direction(glm::vec3(std::numeric_limits<float>::signaling_NaN())),
          reference_frame(atcg::Frame<glm::vec3>()),
          incoming_distance(std::numeric_limits<float>::signaling_NaN()),
          pdf(0.0f)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE bool isValid() const { return !glm::isnan(incoming_distance); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInvalid() { incoming_distance = std::numeric_limits<float>::signaling_NaN(); }

    ATCG_INLINE ATCG_HOST_DEVICE bool isFinite() const { return glm::isfinite(incoming_distance); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInfinite() { incoming_distance = std::numeric_limits<float>::infinity(); }
};

struct DualInteraction
{
    // Output
    CuDiff::Dual<6, glm::vec3> position;
    CuDiff::Dual<6, float> incoming_distance;
    atcg::mat6 dx1x2_dx0x1;
    glm::mat3 dxdw;

    // Input
    CuDiff::Dual<6, glm::vec3> incoming_direction;
    CuDiff::Dual<6, glm::vec3> incoming_position;

    atcg::Frame<glm::vec3> reference_frame;
    float pdf;

    ATCG_HOST_DEVICE DualInteraction()
        : position(CuDiff::Dual<6, glm::vec3>(glm::vec3(std::numeric_limits<float>::signaling_NaN()))),
          incoming_direction(CuDiff::Dual<6, glm::vec3>(glm::vec3(std::numeric_limits<float>::signaling_NaN()))),
          reference_frame(atcg::Frame<glm::vec3>()),
          incoming_distance(CuDiff::Dual<6, float>(std::numeric_limits<float>::signaling_NaN())),
          pdf(0.0f)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE bool isValid() const { return !glm::isnan(incoming_distance.val()); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInvalid()
    {
        incoming_distance = CuDiff::Dual<6, float>(std::numeric_limits<float>::signaling_NaN());
    }

    ATCG_INLINE ATCG_HOST_DEVICE bool isFinite() const { return glm::isfinite(incoming_distance.val()); }

    ATCG_INLINE ATCG_HOST_DEVICE void setInfinite()
    {
        incoming_distance = CuDiff::Dual<6, float>(std::numeric_limits<float>::infinity());
    }
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

struct DualSurfaceInteraction : public DualInteraction
{
    ATCG_HOST_DEVICE DualSurfaceInteraction() : DualInteraction() {}

    // Output
    CuDiff::Dual<6, glm::vec3> normal;
    CuDiff::Dual<6, glm::vec2> uv;
    uint32_t primitive_idx;
    uint32_t entity_id;

    const BSDFVPtrTable *bsdf;
    const EmitterVPtrTable *emitter;
    const MediumVPtrTable *inside_medium;
    const MediumVPtrTable *outside_medium;

    ATCG_DEVICE ATCG_INLINE SurfaceInteraction toSi() const
    {
        SurfaceInteraction si;
        si.position           = position.val();
        si.normal             = normal.val();
        si.uv                 = uv.val();
        si.reference_frame    = reference_frame;
        si.incoming_direction = incoming_direction.val();
        si.incoming_distance  = incoming_distance.val();
        si.primitive_idx      = primitive_idx;
        si.entity_id          = entity_id;
        si.bsdf               = bsdf;
        si.emitter            = emitter;
        si.inside_medium      = inside_medium;
        si.outside_medium     = outside_medium;
        si.pdf                = pdf;
        si.color              = glm::vec3(1);    // TODO

        return si;
    }
};

struct DualMediumInteraction : public DualInteraction
{
    ATCG_HOST_DEVICE DualMediumInteraction() : DualInteraction() {}

    ATCG_DEVICE ATCG_INLINE MediumInteraction toMi() const
    {
        MediumInteraction mi;
        mi.position           = position.val();
        mi.incoming_direction = incoming_direction.val();
        mi.reference_frame    = reference_frame;
        mi.incoming_distance  = incoming_distance.val();
        mi.pdf                = pdf;

        return mi;
    }
};

struct AnyInteraction
{
    enum Type
    {
        InteractionType,
        SurfaceInteractionType,
        MediumInteractionType,
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


struct AnyDualInteraction
{
    enum Type
    {
        DualInteractionType,
        DualSurfaceInteractionType,
        DualMediumInteractionType,
    };
    Type type;

    union
    {
        DualInteraction it;
        DualSurfaceInteraction si;
        DualMediumInteraction mi;
    };

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction() : type(DualInteractionType), it(DualInteraction()) {}

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction(const DualInteraction &interaction)
        : type(DualInteractionType),
          it(interaction)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction(const DualSurfaceInteraction &surface)
        : type(DualSurfaceInteractionType),
          si(surface)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction(const DualMediumInteraction &medium)
        : type(DualMediumInteractionType),
          mi(medium)
    {
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction(const AnyDualInteraction &ai)
    {
        this->type = ai.type;
        switch(ai.type)
        {
            case DualInteractionType:
            {
                this->it = ai.it;
                break;
            }
            case DualMediumInteractionType:
            {
                this->mi = ai.mi;
                break;
            }
            case DualSurfaceInteractionType:
            {
                this->si = ai.si;
                break;
            }
        }
    }

    ATCG_INLINE ATCG_HOST_DEVICE operator DualInteraction &() { return it; }
    ATCG_INLINE ATCG_HOST_DEVICE operator const DualInteraction &() const { return it; }
    ATCG_INLINE ATCG_HOST_DEVICE operator DualSurfaceInteraction &() { return si; }
    ATCG_INLINE ATCG_HOST_DEVICE operator const DualSurfaceInteraction &() const { return si; }
    ATCG_INLINE ATCG_HOST_DEVICE operator DualMediumInteraction &() { return mi; }
    ATCG_INLINE ATCG_HOST_DEVICE operator const DualMediumInteraction &() const { return mi; }

    ATCG_INLINE ATCG_HOST_DEVICE DualInteraction *operator->() { return &it; }
    ATCG_INLINE ATCG_HOST_DEVICE const DualInteraction *operator->() const { return &it; }

    ATCG_INLINE ATCG_HOST_DEVICE bool is_surface() const { return type == DualSurfaceInteractionType; }
    ATCG_INLINE ATCG_HOST_DEVICE bool is_medium() const { return type == DualMediumInteractionType; }

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction &operator=(const DualInteraction &it)
    {
        this->type = DualInteractionType;
        this->it   = it;
        return *this;
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction &operator=(const DualMediumInteraction &mi)
    {
        this->type = DualMediumInteractionType;
        this->mi   = mi;
        return *this;
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction &operator=(const DualSurfaceInteraction &si)
    {
        this->type = DualSurfaceInteractionType;
        this->si   = si;
        return *this;
    }

    ATCG_INLINE ATCG_HOST_DEVICE AnyDualInteraction &operator=(const AnyDualInteraction &ai)
    {
        this->type = ai.type;
        switch(ai.type)
        {
            case DualInteractionType:
            {
                this->it = ai.it;
                break;
            }
            case DualMediumInteractionType:
            {
                this->mi = ai.mi;
                break;
            }
            case DualSurfaceInteractionType:
            {
                this->si = ai.si;
                break;
            }
        }
        return *this;
    }
};

}    // namespace atcg