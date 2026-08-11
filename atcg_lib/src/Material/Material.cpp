#include <Core/Assert.h>
#include <Core/Application.h>
#include <Material/Material.h>
#include <Renderer/Renderer.h>
#include <Renderer/Shader.h>

namespace atcg
{
Material::Material(const std::string& type) : _material_type(type) {}

void Material::releaseTextureIDs(RendererSystem* renderer)
{
    ATCG_ASSERT(_uploaded, "Tried freeing material ids without while material is not uploaded");

    renderer->pushTextureID(_used_texture_ids[0]);
    renderer->pushTextureID(_used_texture_ids[1]);
    renderer->pushTextureID(_used_texture_ids[2]);
    renderer->pushTextureID(_used_texture_ids[3]);
    renderer->pushTextureID(_used_texture_ids[4]);

    _uploaded = false;
}
}    // namespace atcg