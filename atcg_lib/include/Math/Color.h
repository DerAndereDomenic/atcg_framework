#pragma once

#include <Core/CUDA.h>

namespace atcg
{
namespace Color
{
/**
 * @brief Quantize a color 8 bit
 *
 * @param color The hdr color
 *
 * @return The quanitized color
 */
ATCG_HOST_DEVICE inline glm::u8vec3 quantize(const glm::vec3& color);

/**
 * @brief Dequantize a color
 *
 * @param color The ldr color
 *
 * @return The dequanized color
 */
ATCG_HOST_DEVICE inline glm::vec3 dequantize(const glm::u8vec3& color);

/**
 * @brief Quantize a color 8 bit
 *
 * @param color The hdr color
 *
 * @return The quanitized color
 */
ATCG_HOST_DEVICE inline glm::u8vec4 quantize(const glm::vec4& color);

/**
 * @brief Dequantize a color
 *
 * @param color The ldr color
 *
 * @return The dequanized color
 */
ATCG_HOST_DEVICE inline glm::vec4 dequantize(const glm::u8vec4& color);

/**
 * @brief Convert from sRGB to lRGB (linear) using the D65 observer
 *
 * @param color The sRGB color
 *
 * @return The lRGB color
 */
ATCG_HOST_DEVICE inline glm::vec3 sRGB_to_lRGB(const glm::vec3& color);

/**
 * @brief Convert from lRGB (linear) to sRGB using the D65 observer
 *
 * @param color The lRGB color
 *
 * @return The sRGB color
 */
ATCG_HOST_DEVICE inline glm::vec3 lRGB_to_sRGB(const glm::vec3& color);

/**
 * @brief Convert from sRGB to XYZ (D65)
 *
 * @param color The sRGB color
 *
 * @return The XYZ representation
 */
ATCG_HOST_DEVICE inline glm::vec3 sRGB_to_XYZ(const glm::vec3& color);

/**
 * @brief Convert from XYZ to sRGB (D65)
 *
 * @param color The XYZ color
 *
 * @return The sRGB representation
 */
ATCG_HOST_DEVICE inline glm::vec3 XYZ_to_sRGB(const glm::vec3& color);

/**
 * @brief Convert from lRGB to XYZ (D65)
 *
 * @param color The lRGB color
 *
 * @return The XYZ representation
 */
ATCG_HOST_DEVICE inline glm::vec3 lRGB_to_XYZ(const glm::vec3& color);

/**
 * @brief Convert from XYZ to lRGB (D65)
 *
 * @param color The XYZ color
 *
 * @return The lRGB representation
 */
ATCG_HOST_DEVICE inline glm::vec3 XYZ_to_lRGB(const glm::vec3& color);
}    // namespace Color
}    // namespace atcg

#include "../../src/Math/ColorDetail.h"