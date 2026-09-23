#pragma once

#include <Core/Assert.h>
#include <Core/SystemRegistry.h>
#include <DataStructure/Registry.h>
#include <Renderer/RenderPass.h>

#include <functional>

namespace atcg
{
namespace RenderPassRegistry
{
using RenderPassBuilder = std::function<atcg::ref_ptr<RenderPass>(atcg::Dictionary&)>;

struct RenderPassFunctions
{
    RenderPassBuilder builder;
};

using Registry = atcg::Registry<RenderPassFunctions>;

ATCG_API void registerRenderPass(Registry* registry, std::string_view type, RenderPassFunctions functions);

ATCG_API atcg::ref_ptr<RenderPass>
createRenderPass(Registry* registry, const std::string& type, atcg::Dictionary& properties);

ATCG_INLINE Registry* getRegistry()
{
    Registry* registry = SystemRegistry::instance()->getSystem<RenderPassRegistry::Registry>();
    ATCG_ASSERT(registry, "RenderPass registry not found");
    return registry;
}

ATCG_INLINE void registerRenderPass(std::string_view type, RenderPassFunctions functions)
{
    registerRenderPass(getRegistry(), type, std::move(functions));
}

ATCG_INLINE atcg::ref_ptr<RenderPass> createRenderPass(const std::string& type, atcg::Dictionary& properties)
{
    return createRenderPass(getRegistry(), type, properties);
}

}    // namespace RenderPassRegistry
}    // namespace atcg

#define ATCG_REGISTER_RENDER_PASS(registry, RenderPassType, RenderPassClass)                                           \
    {                                                                                                                  \
        atcg::RenderPassRegistry::RenderPassFunctions functions = {                                                    \
            [](atcg::Dictionary& properties) -> atcg::ref_ptr<atcg::RenderPass>                                        \
            { return atcg::make_ref<RenderPassClass>(properties); }};                                                  \
        registry->registerType(RenderPassType, std::move(functions));                                                  \
    }

#define ATCG_REGISTER_RENDER_PASS_PLUGIN(registry, handle, RenderPassType, RenderPassClass)                            \
    {                                                                                                                  \
        atcg::RenderPassRegistry::RenderPassFunctions functions = {                                                    \
            [](atcg::Dictionary& properties) -> atcg::ref_ptr<atcg::RenderPass>                                        \
            { return atcg::make_ref<RenderPassClass>(properties); }};                                                  \
        registry->registerType(handle, RenderPassType, std::move(functions));                                          \
    }