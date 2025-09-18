#pragma once

#include <Core/glm.h>
#include <Core/CUDA.h>

template<typename T>
ATCG_DEVICE ATCG_INLINE T tex3D(cudaTextureObject_t tex, glm::vec3 coord)
{
    if constexpr(std::is_same_v<T, float>)
    {
        return tex3D<float>(tex, coord.x, coord.y, coord.z);
    }
    else if constexpr(std::is_same_v<T, glm::vec2>)
    {
        float4 v = tex3D<float4>(tex, coord.x, coord.y, coord.z);
        return glm::vec2(v.x, v.y);
    }
    else if constexpr(std::is_same_v<T, glm::vec3>)
    {
        float4 v = tex3D<float4>(tex, coord.x, coord.y, coord.z);
        return glm::vec3(v.x, v.y, v.z);
    }
    else
    {
        float4 v = tex3D<float4>(tex, coord.x, coord.y, coord.z);
        return glm::vec4(v.x, v.y, v.z, v.w);
    }
}

template<typename T>
struct GridData
{
    cudaTextureObject_t texture = 0;
    glm::mat4 to_uvw            = glm::mat4(1);
    T default_value             = T(1);
    float scale                 = 1.0f;

#ifdef __CUDACC__
    ATCG_DEVICE ATCG_INLINE bool is_valid() const { return texture != 0; }

    ATCG_DEVICE ATCG_INLINE T eval(const glm::vec3& world_pos) const
    {
        if(!is_valid())
        {
            return default_value;
        }

        glm::vec4 local_pos_hom = to_uvw * glm::vec4(world_pos, 1);
        glm::vec3 local_pos     = glm::xyz(local_pos_hom) / local_pos_hom.w;
        T value                 = tex3D<T>(texture, local_pos);

        return value;
    }

    ATCG_DEVICE ATCG_INLINE glm::vec3 posToLocal(const glm::vec3& world_pos) const
    {
        glm::vec4 local_pos_hom = to_uvw * glm::vec4(world_pos, 1);
        glm::vec3 local_pos     = glm::xyz(local_pos_hom);    // assert w = 1!
        return local_pos;
    }

    ATCG_DEVICE ATCG_INLINE glm::vec3 dirToLocal(const glm::vec3& world_dir) const
    {
        glm::vec4 local_dir_hom = to_uvw * glm::vec4(world_dir, 0);
        glm::vec3 local_dir     = glm::xyz(local_dir_hom);    // assert w=0?
        return local_dir;
    }

    ATCG_DEVICE ATCG_INLINE T evalLocal(const glm::vec3& local_pos) const
    {
        if(!is_valid()) return default_value;

        T value = tex3D<T>(texture, local_pos);
        return value;
    }
#endif
};

namespace atcg
{

struct HeterogeneousMediumData
{
    glm::mat4 world_to_local;
    GridData<glm::vec3> albedo_grid;
    float density_majorant;
    GridData<float> density_grid;
    GridData<glm::vec3> emission_grid;
};

}    // namespace atcg