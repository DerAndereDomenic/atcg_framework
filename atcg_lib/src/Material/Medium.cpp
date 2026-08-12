#include <Material/Medium.h>

#include <Core/Assert.h>

namespace atcg
{
Medium::Medium(const std::string& type, const atcg::Dictionary& dict) : _medium_type(type) {}

void Medium::releaseTextureIDs(RendererSystem* renderer)
{
    ATCG_ASSERT(_uploaded, "Tried freeing material ids without while material is not uploaded");

    renderer->pushTextureID(_texture_ids[0]);
    renderer->pushTextureID(_texture_ids[1]);
    renderer->pushTextureID(_texture_ids[2]);

    _uploaded = false;
}

}    // namespace atcg