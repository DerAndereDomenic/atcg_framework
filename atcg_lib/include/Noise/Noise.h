#pragma once

#include <Core/Memory.h>
#include <Renderer/Texture.h>
#include <glm/glm.hpp>

namespace atcg
{
namespace Noise
{
// TODO:
// atcg::ref_ptr<Texture2D> createWhiteNoiseTexture1D(uint32_t dim);

/**
 * @brief Create a 2D texture with white noise.
 * The floating point texture will have uniformly sampled pixels between [0,1]
 *
 * @param dim The dimensions of the texture
 * @return The noise texture
 */
atcg::ref_ptr<Texture2D> createWhiteNoiseTexture2D(glm::ivec2 dim);

// TODO:
// atcg::ref_ptr<Texture2D> createWhiteNoiseTexture3D(glm::ivec3 dim);
}    // namespace Noise
}    // namespace atcg