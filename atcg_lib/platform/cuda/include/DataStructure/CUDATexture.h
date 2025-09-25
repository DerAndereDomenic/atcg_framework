#pragma once

#include <Core/CUDA.h>
#include <Renderer/TextureSpecification.h>

namespace atcg
{
/**
 * @brief A class to model a texture in CUDA for read and write
 *
 * @tparam T The internal texture type (float or a glm::vec type)
 */
template<typename T>
struct CUDATexture
{
    /**
     * @brief Constructor
     */
    CUDATexture() = default;

    /**
     * @brief Read the texture
     *
     * @param uv The uv coordinates
     * @return The data as normalized float
     */
    ATCG_DEVICE
    glm::vec4 read(const glm::vec2& uv);

    /**
     * @brief Write to the texture
     *
     * @param val The value to write
     * @param texel The texel (int)
     */
    ATCG_DEVICE
    void write(const T& val, const glm::ivec2& texel);

    /**
     * @brief Read the texture
     *
     * @param uvw The uv coordinates
     * @return The data as normalized float
     */
    ATCG_DEVICE
    glm::vec4 read(const glm::vec3& uvw);

    /**
     * @brief Write to the texture
     *
     * @param val The value to write
     * @param texel The texel (int)
     */
    ATCG_DEVICE
    void write(const T& val, const glm::ivec3& texel);

    // Texture and Surface Object
    struct
    {
        cudaTextureObject_t texture = 0;
        cudaSurfaceObject_t surface = 0;
    } texture_data = {};

    // Texture Specification
    TextureSpecification spec;

    // Default value to read
    glm::vec4 default_value = glm::vec4(0);
};

// Implementation

template<typename T>
ATCG_DEVICE glm::vec4 CUDATexture<T>::read(const glm::vec2& uv)
{
    if(texture_data.surface != 0)
    {
        // Read using cuda api
        if constexpr(std::is_same_v<T, float>)
        {
            return glm::vec4(tex2D<float>(texture_data.texture, uv.x, uv.y));
        }
        else
        {
            return cuda2glm(tex2D<float4>(texture_data.texture, uv.x, uv.y));
        }
    }

    // Else return default value
    return default_value;
}

template<typename T>
ATCG_DEVICE void CUDATexture<T>::write(const T& val, const glm::ivec2& texel)
{
    // Assert T fits spec?
    if(texture_data.surface != 0)
    {
        // Write using cuda api
        auto cuda_val = glm2cuda(val);
        if constexpr(std::is_same_v<T, float>)
        {
            surf2Dwrite(cuda_val, texture_data.surface, texel.x * sizeof(float), texel.y);
        }
        else if constexpr(inv_cuda_type<decltype(cuda_val)>::dim == 3)
        {
            auto cuda_val_pad = make_cuda_type<4, typename T::value_type>::apply(val.x, val.y, val.z, 0);
            surf2Dwrite(cuda_val_pad, texture_data.surface, texel.x * sizeof(decltype(cuda_val_pad)), texel.y);
        }
        else
        {
            surf2Dwrite(cuda_val, texture_data.surface, texel.x * sizeof(decltype(cuda_val)), texel.y);
        }
    }
    // Else do nothing (no valid data)
}

template<typename T>
ATCG_DEVICE glm::vec4 CUDATexture<T>::read(const glm::vec3& uvw)
{
    if(texture_data.surface != 0)
    {
        // Read using cuda api
        if constexpr(std::is_same_v<T, float>)
        {
            return glm::vec4(tex3D<float>(texture_data.texture, uvw.x, uvw.y, uvw.z));
        }
        else
        {
            return cuda2glm(tex3D<float4>(texture_data.texture, uvw.x, uvw.y, uvw.z));
        }
    }

    // Else return default value
    return default_value;
}

template<typename T>
ATCG_DEVICE void CUDATexture<T>::write(const T& val, const glm::ivec3& texel)
{
    // Assert T fits spec?
    if(texture_data.surface != 0)
    {
        // Write using cuda api
        auto cuda_val = glm2cuda(val);
        if constexpr(std::is_same_v<T, float>)
        {
            surf3Dwrite(cuda_val, texture_data.surface, texel.x * sizeof(float), texel.y, texel.z);
        }
        else if constexpr(inv_cuda_type<decltype(cuda_val)>::dim == 3)
        {
            auto cuda_val_pad = make_cuda_type<4, typename T::value_type>::apply(val.x, val.y, val.z, 0);
            surf3Dwrite(cuda_val_pad, texture_data.surface, texel.x * sizeof(decltype(cuda_val_pad)), texel.y, texel.z);
        }
        else
        {
            surf3Dwrite(cuda_val, texture_data.surface, texel.x * sizeof(decltype(cuda_val)), texel.y, texel.z);
        }
    }

    // Else do nothing (no valid data)
}
}    // namespace atcg