#pragma once

#include <Core/OptixComponent.h>
#include <Core/glm.h>
#include <Shape/Shape.h>

namespace atcg
{
class ShapeInstance : public OptixComponent
{
public:
    ShapeInstance(const atcg::ref_ptr<Shape>& shape, const glm::mat4& transform);

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    ATCG_INLINE atcg::ref_ptr<Shape> getShape() const { return _shape; }

    ATCG_INLINE const glm::mat4& getTransform() const { return _transform; }

private:
    glm::mat4 _transform;
    atcg::ref_ptr<Shape> _shape;
};
}    // namespace atcg