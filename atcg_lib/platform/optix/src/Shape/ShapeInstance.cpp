#include <Shape/ShapeInstance.h>
#include <Shape/ShapeInstanceData.cuh>

namespace atcg
{
ShapeInstance::ShapeInstance(const Dictionary& shape_data)
{
    _shape          = shape_data.getValueOr<atcg::ref_ptr<Shape>>("shape", nullptr);
    _bsdf           = shape_data.getValueOr<atcg::ref_ptr<BSDF>>("bsdf", nullptr);
    _emitter        = shape_data.getValueOr<atcg::ref_ptr<Emitter>>("emitter", nullptr);
    _inside_medium  = shape_data.getValueOr<atcg::ref_ptr<Medium>>("inside_medium", nullptr);
    _outside_medium = shape_data.getValueOr<atcg::ref_ptr<Medium>>("outside_medium", nullptr);
    _transform      = shape_data.getValueOr<glm::mat4>("transform", glm::mat4(1));
    _entity_id      = shape_data.getValueOr<int32_t>("entity_id", -1);
    _color          = shape_data.getValueOr<glm::vec3>("color", glm::vec3(1));
}

void ShapeInstance::onImGuiRender()
{
    if(_shape) _shape->onImGuiRender();
    if(_bsdf) _bsdf->onImGuiRender();
    if(_emitter) _emitter->onImGuiRender();
    if(_inside_medium) _inside_medium->onImGuiRender();
    if(_outside_medium) _outside_medium->onImGuiRender();
}

void ShapeInstance::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(!_shape) return;
    _shape->ensureInitialized(pipeline, sbt);
    if(_emitter) _emitter->ensureInitialized(pipeline, sbt);
    if(_inside_medium) _inside_medium->ensureInitialized(pipeline, sbt);
    if(_outside_medium) _outside_medium->ensureInitialized(pipeline, sbt);

    ShapeInstanceData data;
    data.shape          = _shape->_shape_data;
    data.bsdf           = _bsdf ? _bsdf->getVPtrTable() : nullptr;
    data.emitter        = _emitter ? _emitter->getVPtrTable() : nullptr;
    data.inside_medium  = _inside_medium ? _inside_medium->getVPtrTable() : nullptr;
    data.outside_medium = _outside_medium ? _outside_medium->getVPtrTable() : nullptr;
    data.entity_id      = _entity_id;
    data.color          = _color;
    sbt->addHitEntry(_shape->getHitGroup(), data);
}
}    // namespace atcg