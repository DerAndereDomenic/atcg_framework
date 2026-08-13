#pragma once

#include <Core/RaytracingComponent.h>
#include <Core/glm.h>
#include <DataStructure/Dictionary.h>
#include <Shape/Shape.h>
#include <Material/Material.h>
#include <Emitter/Emitter.h>
#include <Material/Medium.h>

namespace atcg
{
/**
 * @brief Class to model a shape instance
 */
class ATCG_API ShapeInstance : public RaytracingComponent
{
public:
    /**
     * @brief Constructor from a dictionary.
     * The dictionary expects
     * - shape: atcg::ref_ptr<Shape>
     * - bsdf: atcg::ref_ptr<BSDF>
     * - transform: glm::mat4
     * - entity_id: int32_t
     * - color: glm::vec3
     * - emitter: atcg::ref_ptr<Emitter>
     *
     * @param shape_data The shape data
     */
    ShapeInstance(const Dictionary& shape_data);

    virtual ~ShapeInstance() = default;

    // TODO
    virtual void updateData() override {}

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    /**
     * @brief Get the shape
     *
     * @return The shape
     */
    ATCG_INLINE atcg::ref_ptr<Shape> getShape() const { return _shape; }

    ATCG_INLINE atcg::ref_ptr<Material> getMaterial() const { return _material; }

    ATCG_INLINE atcg::ref_ptr<Emitter> getEmitter() const { return _emitter; }

    ATCG_INLINE atcg::ref_ptr<Medium> getInsideMedium() const { return _inside_medium; }

    ATCG_INLINE atcg::ref_ptr<Medium> getOutsideMedium() const { return _outside_medium; }

    /**
     * @brief Get the transform
     *
     * @return The transform
     */
    ATCG_INLINE const glm::mat4& getTransform() const { return _transform; }

    ATCG_INLINE uint32_t entity_id() const { return _entity_id; }

    ATCG_INLINE glm::vec3 color() const { return _color; }

private:
    glm::mat4 _transform;
    atcg::ref_ptr<Shape> _shape;
    atcg::ref_ptr<Material> _material;
    atcg::ref_ptr<Emitter> _emitter;
    atcg::ref_ptr<Medium> _inside_medium;
    atcg::ref_ptr<Medium> _outside_medium;
    uint32_t _entity_id;
    glm::vec3 _color;
};
}    // namespace atcg