#pragma once

#include <Math/Constants.h>
#include <Math/Utils.h>

namespace atcg
{
namespace Color
{

namespace detail
{
static ATCG_CONSTEXPR glm::mat3 lRGB_to_XYZ_transform(glm::vec3(0.4124f, 0.2126f, 0.0193f),
                                                      glm::vec3(0.3576f, 0.7152f, 0.1192f),
                                                      glm::vec3(0.1805f, 0.0722f, 0.9505f));

static ATCG_CONSTEXPR glm::mat3 XYZ_to_lRBB_transform(glm::vec3(+3.2406f, -0.9689, +0.0557f),
                                                      glm::vec3(-1.5372f, +1.8758f, -0.2040f),
                                                      glm::vec3(-0.4986f, +0.0415f, +1.0570f));

ATCG_HOST_DEVICE ATCG_FORCE_INLINE uint8_t quantize_channel(const float c)
{
    return (uint8_t)(c * 255.0f);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float dequantize_channel(const uint8_t c)
{
    return (float)(c) / 255.0f;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float sRGB_to_lRGB_channel(const float c)
{
    return c <= 0.04045f ? c / 12.92f : glm::pow((c + 0.055f) / 1.055f, 2.4f);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float lRGB_to_sRGB_channel(const float c)
{
    return c <= 0.0031308 ? 12.92f * c : 1.055f * glm::pow(c, 1.0f / 2.4f) - 0.055f;
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T g(const T x, const T mu, const T t1, const T t2)
{
    return x < mu ? glm::exp(-t1 * t1 * (x - mu) * (x - mu) / 2) : glm::exp(-t2 * t2 * (x - mu) * (x - mu) / 2);
}

/**
 * @brief Read from an array and linear interpolate between neighboring entries.
 *
 * @tparam T The datatype
 * @param data The array data
 * @param size The array size
 * @param index The float index
 *
 * @return The interpolated value
 */
template<typename T>
ATCG_HOST_DEVICE T read_array_interpolated(const T* data, const uint32_t size, const float index)
{
    T idx              = glm::clamp(index, T(0), T(size - 1));
    uint32_t bin_index = (uint32_t)idx;
    T fract_index      = idx - T(bin_index);    // glm::fract

    return bin_index < size - 1 ? (T(1.0) - fract_index) * data[bin_index] + fract_index * data[bin_index + 1]
                                : (T(1.0) + fract_index) * data[bin_index] - fract_index * data[bin_index - 1];
}
}    // namespace detail

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::u8vec3 quantize(const glm::vec3& color)
{
    return glm::u8vec3(detail::quantize_channel(color.x),
                       detail::quantize_channel(color.y),
                       detail::quantize_channel(color.z));
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 dequantize(const glm::u8vec3& color)
{
    return glm::vec3(detail::dequantize_channel(color.x),
                     detail::dequantize_channel(color.y),
                     detail::dequantize_channel(color.z));
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::u8vec4 quantize(const glm::vec4& color)
{
    return glm::u8vec4(quantize(glm::vec3(color)), detail::quantize_channel(color.a));
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec4 dequantize(const glm::u8vec4& color)
{
    return glm::vec4(dequantize(glm::vec3(color)), detail::dequantize_channel(color.a));
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 sRGB_to_lRGB(const glm::vec3& color)
{
    return glm::vec3(detail::sRGB_to_lRGB_channel(color.x),
                     detail::sRGB_to_lRGB_channel(color.y),
                     detail::sRGB_to_lRGB_channel(color.z));
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 lRGB_to_sRGB(const glm::vec3& color)
{
    return glm::vec3(detail::lRGB_to_sRGB_channel(color.x),
                     detail::lRGB_to_sRGB_channel(color.y),
                     detail::lRGB_to_sRGB_channel(color.z));
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 sRGB_to_XYZ(const glm::vec3& color)
{
    glm::vec3 lRGB = sRGB_to_lRGB(color);
    return lRGB_to_XYZ(lRGB);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 XYZ_to_sRGB(const glm::vec3& color)
{
    glm::vec3 lRGB = XYZ_to_lRGB(color);
    return lRGB_to_sRGB(lRGB);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 lRGB_to_XYZ(const glm::vec3& color)
{
    return detail::lRGB_to_XYZ_transform * color;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE glm::vec3 XYZ_to_lRGB(const glm::vec3& color)
{
    return detail::XYZ_to_lRBB_transform * color;
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float lRGB_to_luminance(const glm::vec3& color)
{
    const glm::vec3 luminance_weights = glm::row(detail::lRGB_to_XYZ_transform, 1);
    return glm::dot(luminance_weights, color);
}

ATCG_HOST_DEVICE ATCG_FORCE_INLINE float sRGB_to_luminance(const glm::vec3& color)
{
    auto lRGB = sRGB_to_lRGB(color);
    return lRGB_to_luminance(lRGB);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T color_matching_x(const T lambda)
{
    return T(1.056) * detail::g(lambda, T(599.8), T(0.0264), T(0.0323)) +
           T(0.362) * detail::g(lambda, T(442.0), T(0.0624), T(0.0374)) -
           T(0.065) * detail::g(lambda, T(501.1), T(0.0490), T(0.0382));
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T color_matching_y(const T lambda)
{
    return T(0.821) * detail::g(lambda, T(568.8), T(0.0213), T(0.0247)) +
           T(0.286) * detail::g(lambda, T(530.9), T(0.0613), T(0.0322));
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T color_matching_z(const T lambda)
{
    return T(1.217) * detail::g(lambda, T(437.0), T(0.0845), T(0.0278)) +
           T(0.681) * detail::g(lambda, T(459.0), T(0.0385), T(0.0725));
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T planck(const T lambda, const T temperature)
{
    T lambda_m      = lambda * T(1e-9);
    constexpr T c_2 = Constants::c<T>() * Constants::c<T>();
    T num           = T(2.0) * Constants::h<T>() * c_2;
    T lambda_m_5    = lambda_m * lambda_m * lambda_m * lambda_m * lambda_m;
    T denom =
        lambda_m_5 *
        (glm::exp((Constants::h<T>() * Constants::c<T>()) / (lambda_m * Constants::boltzmann<T>() * temperature)) -
         T(1.0));

    return num / denom;
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T D65(const T lambda)
{
    float index = (lambda - atcg::Spectral::D65_MIN_LAMBDA) /
                  (atcg::Spectral::D65_MAX_LAMBDA - atcg::Spectral::D65_MIN_LAMBDA) * atcg::Spectral::D65_SIZE;
    return detail::read_array_interpolated(atcg::Spectral::D65_data, atcg::Spectral::D65_SIZE, index);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T Sr(const T lambda)
{
    float index = (lambda - atcg::Spectral::SPECTRAL_BASIS_MIN_LAMBDA) /
                  (atcg::Spectral::SPECTRAL_BASIS_MAX_LAMBDA - atcg::Spectral::SPECTRAL_BASIS_MIN_LAMBDA) *
                  atcg::Spectral::SPECTRAL_BASIS_SIZE;
    return detail::read_array_interpolated(atcg::Spectral::basis_Sr_data, atcg::Spectral::SPECTRAL_BASIS_SIZE, index);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T Sg(const T lambda)
{
    float index = (lambda - atcg::Spectral::SPECTRAL_BASIS_MIN_LAMBDA) /
                  (atcg::Spectral::SPECTRAL_BASIS_MAX_LAMBDA - atcg::Spectral::SPECTRAL_BASIS_MIN_LAMBDA) *
                  atcg::Spectral::SPECTRAL_BASIS_SIZE;
    return detail::read_array_interpolated(atcg::Spectral::basis_Sg_data, atcg::Spectral::SPECTRAL_BASIS_SIZE, index);
}

template<typename T>
ATCG_HOST_DEVICE ATCG_FORCE_INLINE T Sb(const T lambda)
{
    float index = (lambda - atcg::Spectral::SPECTRAL_BASIS_MIN_LAMBDA) /
                  (atcg::Spectral::SPECTRAL_BASIS_MAX_LAMBDA - atcg::Spectral::SPECTRAL_BASIS_MIN_LAMBDA) *
                  atcg::Spectral::SPECTRAL_BASIS_SIZE;
    return detail::read_array_interpolated(atcg::Spectral::basis_Sb_data, atcg::Spectral::SPECTRAL_BASIS_SIZE, index);
}
}    // namespace Color
}    // namespace atcg