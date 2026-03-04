#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>

#include <DataStructure/TextureSampler.h>

template<typename T>
struct CUDATextureStorage
{
    cudaTextureObject_t texture = 0;
#ifdef __CUDACC__
    ATCG_DEVICE ATCG_INLINE T eval(const glm::vec3& coord) const
    {
        if constexpr(std::is_same_v<T, float>)
        {
            return tex3D<float>(texture, coord.x, coord.y, coord.z);
        }
        else if constexpr(std::is_same_v<T, glm::vec2>)
        {
            float4 v = tex3D<float4>(texture, coord.x, coord.y, coord.z);
            return glm::vec2(v.x, v.y);
        }
        else if constexpr(std::is_same_v<T, glm::vec3>)
        {
            float4 v = tex3D<float4>(texture, coord.x, coord.y, coord.z);
            return glm::vec3(v.x, v.y, v.z);
        }
        else
        {
            float4 v = tex3D<float4>(texture, coord.x, coord.y, coord.z);
            return glm::vec4(v.x, v.y, v.z, v.w);
        }
    }
#endif

    ATCG_INLINE ATCG_DEVICE bool is_valid() const { return texture != 0; }
};

template<typename T>
struct TextureSamplerStorage
{
    atcg::TextureSampler<T> sampler;

    ATCG_INLINE ATCG_DEVICE bool is_valid() const
    {
        return sampler.getSpecification().width > 0 && sampler.getSpecification().height > 0;
    }

    ATCG_INLINE ATCG_DEVICE T eval(const glm::vec3& uv) const { return sampler.read(uv); }
};

template<typename T, typename TextureStorageType>
struct GridData
{
    TextureStorageType storage;
    glm::mat4 to_uvw = glm::mat4(1);
    T default_value  = T(1);
    float scale      = 1.0f;

#ifdef __CUDACC__
    ATCG_DEVICE ATCG_INLINE bool is_valid() const { return storage.is_valid(); }

    ATCG_DEVICE ATCG_INLINE T eval(const glm::vec3& world_pos) const
    {
        if(!is_valid())
        {
            return scale * default_value;
        }

        glm::vec4 local_pos_hom = to_uvw * glm::vec4(world_pos, 1);
        glm::vec3 local_pos     = glm::xyz(local_pos_hom) / local_pos_hom.w;
        T value                 = storage.eval(local_pos);

        return scale * value;
    }
#endif
};

namespace atcg
{

struct HeterogeneousMediumData
{
    GridData<glm::vec3, CUDATextureStorage<glm::vec3>> albedo_grid;
    float density_majorant;
    GridData<float, CUDATextureStorage<float>> density_grid;
    GridData<glm::vec3, CUDATextureStorage<glm::vec3>> emission_grid;

    TextureSampler<float> density_sampler;
};

}    // namespace atcg