#include <BSDF/BSDF.h>
#include <Medium/Medium.h>
#include <Shape/Shape.h>
#include <Emitter/Emitter.h>
#include <Scene/ComponentRegistry.h>

namespace atcg
{
namespace GUI
{
ATCG_REGISTER_COMPONENT_DRAW(BSDFComponent);
ATCG_REGISTER_COMPONENT_DRAW(MediumComponent);
ATCG_REGISTER_COMPONENT_DRAW(EmitterComponent);
ATCG_REGISTER_COMPONENT_DRAW(ShapeComponent);
}    // namespace GUI
}    // namespace atcg
