#include <Renderer/RenderPassRegistry.h>

namespace atcg
{
namespace RenderPassRegistry
{
void registerRenderPass(Registry* registry, std::string_view type, RenderPassFunctions functions)
{
    registry->registerType(type, std::move(functions));
}

atcg::ref_ptr<RenderPass> createRenderPass(Registry* registry, const std::string& type, atcg::Dictionary& properties)
{
    const RenderPassFunctions* functions = registry->find(type);
    if(!functions)
    {
        ATCG_ERROR("RenderPass type {} not found in registry", type);
        return nullptr;
    }
    return functions->builder(properties);
}
}    // namespace RenderPassRegistry
}    // namespace atcg