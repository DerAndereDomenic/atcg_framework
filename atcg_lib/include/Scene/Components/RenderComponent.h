#pragma once

#include <Renderer/Material.h>

namespace atcg
{
struct ATCG_API RenderComponent
{
    RenderComponent() { default_material = atcg::make_ref<OpaqueMaterial>(); }

    bool visible = true;

    atcg::ref_ptr<Material> default_material;
};
}    // namespace atcg