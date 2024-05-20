#pragma once

#include <Core/CUDA.h>
#include <Math/SpectrumData.h>

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

/**
 * @brief Get the luminace (Y) of the lRGB color
 *
 * @param color The lRGB color
 *
 * @return The luminance
 */
ATCG_HOST_DEVICE inline float lRGB_to_luminance(const glm::vec3& color);

/**
 * @brief Get the luminace (Y) of the sRGB color
 *
 * @param color The sRGB color
 *
 * @return The luminance
 */
ATCG_HOST_DEVICE inline float sRGB_to_luminance(const glm::vec3& color);

/**
 * @brief The color matching function x_bar.
 * Uses the analytical approximation (https://en.wikipedia.org/wiki/CIE_1931_color_space#Color_matching_functions)
 *
 * @tparam T The type
 * @param lambda The wavelength in nm
 *
 * @return x_bar(lambda)
 */
template<typename T>
ATCG_HOST_DEVICE inline T color_matching_x(const T lambda);

/**
 * @brief The color matching function y_bar.
 * Uses the analytical approximation (https://en.wikipedia.org/wiki/CIE_1931_color_space#Color_matching_functions)
 *
 * @tparam T The type
 * @param lambda The wavelength in nm
 *
 * @return y_bar(lambda)
 */
template<typename T>
ATCG_HOST_DEVICE inline T color_matching_y(const T lambda);

/**
 * @brief The color matching function z_bar.
 * Uses the analytical approximation (https://en.wikipedia.org/wiki/CIE_1931_color_space#Color_matching_functions)
 *
 * @tparam T The type
 * @param lambda The wavelength in nm
 *
 * @return z_bar(lambda)
 */
template<typename T>
ATCG_HOST_DEVICE inline T color_matching_z(const T lambda);

/**
 * @brief Get the planck spectrum for a given wavelength and temperature
 *
 * @tparam T The type
 * @param lambda The wavelength (in nm)
 * @param temperature The temperature (in K)
 *
 * @return The planck spectrum
 */
template<typename T>
ATCG_HOST_DEVICE inline T planck(const T lambda, const T temperature);

/**
 * @brief Evalute the D65 standard illuminant function.
 * Values have to be between 300nm and 780nm.
 * The spectral data is normalized s.t. the discrete version of D65(560) = 100. However, due to interpolation there
 * might be deviations from this value.
 *
 * @tparam T The type
 * @param lambda The wavelength in nm
 *
 * @return The spectral radiance
 */
template<typename T>
ATCG_HOST_DEVICE inline T D65(const T lambda);

/**
 * @brief The red basis function for spectral upsampling from "Spectral Primary Decomposition for Rendering with sRGB
 * Reflectance" by Mallett et al, 2019
 *
 * Wavelength between 380nm and 780nm
 *
 * @tparam T The data type
 * @param lambda The wavelength in nm
 *
 * @return The spectral weight for the red channel
 */
template<typename T>
ATCG_HOST_DEVICE inline T Sr(const T lambda);

/**
 * @brief The green basis function for spectral upsampling from "Spectral Primary Decomposition for Rendering with sRGB
 * Reflectance" by Mallett et al, 2019
 *
 * Wavelength between 380nm and 780nm
 *
 * @tparam T The data type
 * @param lambda The wavelength in nm
 *
 * @return The spectral weight for the green channel
 */
template<typename T>
ATCG_HOST_DEVICE inline T Sg(const T lambda);

/**
 * @brief The blue basis function for spectral upsampling from "Spectral Primary Decomposition for Rendering with sRGB
 * Reflectance" by Mallett et al, 2019
 *
 * Wavelength between 380nm and 780nm
 *
 * @tparam T The data type
 * @param lambda The wavelength in nm
 *
 * @return The spectral weight for the blue channel
 */
template<typename T>
ATCG_HOST_DEVICE inline T Sb(const T lambda);
}    // namespace Color
}    // namespace atcg

#include "../../src/Math/ColorDetail.h"