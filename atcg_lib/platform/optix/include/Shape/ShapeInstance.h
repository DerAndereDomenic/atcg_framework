#pragma once

#include <Core/OptixComponent.h>
#include <Core/glm.h>
#include <DataStructure/Dictionary.h>
#include <Shape/Shape.h>
#include <BSDF/BSDF.h>
#include <Emitter/Emitter.h>

namespace atcg
{
/**
 * @brief Class to model a shape instance
 */
class ShapeInstance : public OptixComponent
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

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
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

    /**
     * @brief Get the transform
     *
     * @return The transform
     */
    ATCG_INLINE const glm::mat4& getTransform() const { return _transform; }

private:
    glm::mat4 _transform;
    atcg::ref_ptr<Shape> _shape;
    atcg::ref_ptr<BSDF> _bsdf;
    atcg::ref_ptr<Emitter> _emitter;
    uint32_t _entity_id;
    glm::vec3 _color;
};
}    // namespace atcg