#include <Shape/ShapeInstance.h>
#include <Shape/ShapeInstanceData.cuh>

namespace atcg
{
ShapeInstance::ShapeInstance(const Dictionary& shape_data)
{
    _shape          = shape_data.getValueOr<atcg::ref_ptr<Shape>>("shape", nullptr);
    _material       = shape_data.getValueOr<atcg::ref_ptr<Material>>("bsdf", nullptr);
    _emitter        = shape_data.getValueOr<atcg::ref_ptr<Emitter>>("emitter", nullptr);
    _inside_medium  = shape_data.getValueOr<atcg::ref_ptr<Medium>>("inside_medium", nullptr);
    _outside_medium = shape_data.getValueOr<atcg::ref_ptr<Medium>>("outside_medium", nullptr);
    _transform      = shape_data.getValueOr<glm::mat4>("transform", glm::mat4(1));
    _entity_id      = shape_data.getValueOr<int32_t>("entity_id", -1);
    _color          = shape_data.getValueOr<glm::vec3>("color", glm::vec3(1));
}

void ShapeInstance::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    auto shape          = getShape();
    auto material       = getMaterial();
    auto emitter        = getEmitter();
    auto inside_medium  = getInsideMedium();
    auto outside_medium = getOutsideMedium();
    if(!shape) return;

    _shape->ensureInitialized(pipeline, sbt);
    if(emitter) _emitter->ensureInitialized(pipeline, sbt);
    if(inside_medium) _inside_medium->ensureInitialized(pipeline, sbt);
    if(outside_medium) _outside_medium->ensureInitialized(pipeline, sbt);

    ShapeInstanceData data;
    data.shape             = shape->getShapeData();
    data.bsdf              = material ? material->getVPtrTable() : nullptr;
    data.emitter           = emitter ? emitter->getVPtrTable() : nullptr;
    data.inside_medium     = inside_medium ? inside_medium->getVPtrTable() : nullptr;
    data.outside_medium    = outside_medium ? outside_medium->getVPtrTable() : nullptr;
    data.entity_id         = entity_id();
    data.color             = color();
    data.object_to_world   = getTransform();
    data.world_to_object   = glm::inverse(getTransform());
    const auto& hit_groups = pipeline->getRayProgramGroups(shape->getShapeType());

    for(const auto& shape_hit_group: hit_groups)
        sbt->addHitEntry(shape_hit_group, data);

    markInitialized();
}
}    // namespace atcg