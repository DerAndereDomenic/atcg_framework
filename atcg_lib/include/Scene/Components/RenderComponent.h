#pragma once

#include <Material/Material.h>
#include <Asset/AssetManagerSystem.h>

namespace atcg
{
struct ATCG_API RenderComponent
{
    RenderComponent() { default_material = AssetManager::getDefaultMaterial(); }

    bool visible = true;

    atcg::ref_ptr<Material> default_material;
};
}    // namespace atcg