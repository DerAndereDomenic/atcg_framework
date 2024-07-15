#pragma once

#include <Math/Random.h>
#include <Pathtracing/SurfaceInteraction.h>
#include <Pathtracing/BSDFFLags.h>
#include <Pathtracing/BSDF.h>

#ifdef ATCG_CUDA_BACKEND
    #include <optix.h>
#endif

namespace atcg
{

struct PBRBSDFData
{
    cudaTextureObject_t diffuse_texture;
    cudaTextureObject_t metallic_texture;
    cudaTextureObject_t roughness_texture;
};

struct RefractiveBSDFData
{
    cudaTextureObject_t diffuse_texture;
    float ior;
};

struct BSDFVPtrTable
{
    uint32_t sampleCallIndex;
    uint32_t evalCallIndex;

    BSDFComponentType flags;

#ifdef __CUDACC__

    __device__ BSDFSamplingResult sampleBSDF(const SurfaceInteraction& si, PCG32& rng) const
    {
        return optixDirectCall<BSDFSamplingResult, const SurfaceInteraction&, PCG32&>(sampleCallIndex, si, rng);
    }

    __device__ BSDFEvalResult evalBSDF(const SurfaceInteraction& si, const glm::vec3& outgoing_dir) const
    {
        return optixDirectCall<BSDFEvalResult, const SurfaceInteraction&, const glm::vec3&>(evalCallIndex,
                                                                                            si,
                                                                                            outgoing_dir);
    }

#endif
};

}    // namespace atcg