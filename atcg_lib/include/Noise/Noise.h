#pragma once

#include <Core/Memory.h>
#include <Core/glm.h>
#include <Renderer/Texture.h>

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

/**
 * @brief Create a 3D texture with white noise.
 * The floating point texture will have uniformly sampled pixels between [0,1]
 *
 * @param dim The dimensions of the texture
 * @return The noise texture
 */
atcg::ref_ptr<Texture3D> createWhiteNoiseTexture3D(glm::ivec3 dim);

/**
 * @brief Create a 2D texture with worley noise.
 * The floating point texture will be unnormalized
 *
 * @param dim The dimensions of the texture
 * @return The noise texture
 */
atcg::ref_ptr<Texture2D> createWorleyNoiseTexture2D(glm::ivec2 dim, uint32_t num_points);

/**
 * @brief Create a 3D texture with worley noise.
 * The floating point texture will be unnormalized
 *
 * @param dim The dimensions of the texture
 * @return The noise texture
 */
atcg::ref_ptr<Texture3D> createWorleyNoiseTexture3D(glm::ivec3 dim, uint32_t num_points);

}    // namespace Noise
}    // namespace atcg