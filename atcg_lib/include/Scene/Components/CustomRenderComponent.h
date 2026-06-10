#pragma once

#include <Renderer/Camera.h>
#include <Scene/Components/RenderComponent.h>
#include <Scene/Entity.h>

namespace atcg
{
struct ATCG_API CustomRenderComponent : public RenderComponent
{
    using RenderCallbackFn = std::function<void(Entity, const atcg::ref_ptr<Camera>& camera)>;

    CustomRenderComponent(const RenderCallbackFn& callback) : RenderComponent(), callback(callback) {}

    RenderCallbackFn callback;
};
}    // namespace atcg