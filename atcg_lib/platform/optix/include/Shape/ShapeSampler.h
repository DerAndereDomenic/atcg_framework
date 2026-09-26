#pragma once

#include <Shape/Shape.h>
#include <DataStructure/Graph.h>
#include <DataStructure/TorchUtils.h>
#include <Shape/ShapeSamplerVPtrTable.cuh>

namespace atcg
{
class ShapeSampler : public OptixComponent
{
public:
    ShapeSampler(const atcg::Dictionary& dict)
    {
        _transform = dict.getValueOr<glm::mat4>("transform", glm::mat4(1));
        _shape     = dict.getValueOr<atcg::ref_ptr<Shape>>("shape", nullptr);
    }

    virtual ~ShapeSampler() {}

    virtual void onImGuiRender() = 0;

    ATCG_INLINE const ShapeSamplerVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    glm::mat4 _transform;
    atcg::ref_ptr<Shape> _shape;

    atcg::dref_ptr<ShapeSamplerVPtrTable> _vptr_table;
};
}    // namespace atcg