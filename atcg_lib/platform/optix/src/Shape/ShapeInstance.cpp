#include <Shape/ShapeInstance.h>
#include <Shape/ShapeInstanceData.cuh>

namespace atcg
{
ShapeInstance::ShapeInstance(const atcg::ref_ptr<Shape>& shape,
                             const atcg::ref_ptr<BSDF>& bsdf,
                             const glm::mat4& transform)
    : _shape(shape),
      _bsdf(bsdf),
      _transform(transform)
{
}

void ShapeInstance::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    _shape->ensureInitialized(pipeline, sbt);

    ShapeInstanceData data;
    data.shape = _shape->_shape_data;
    data.bsdf  = _bsdf->getVPtrTable();
    sbt->addHitEntry(_shape->getHitGroup(), data);
}
}    // namespace atcg