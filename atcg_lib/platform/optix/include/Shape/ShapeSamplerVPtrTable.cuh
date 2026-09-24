#pragma once

#include <Core/glm.h>
#include <Core/SurfaceInteraction.h>
#include <Math/Random.h>

#include <optix.h>

namespace atcg
{

struct ShapeSampleResult
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 uvs;
    float pdf_dA;
};

struct EdgeSampleResult
{
    glm::vec3 position;
    glm::vec3 normal_left;
    glm::vec3 normal_right;
    bool valid_left  = false;
    bool valid_right = false;
    glm::vec3 tangent;
    float pdf_dl;

    ATCG_DEVICE ATCG_INLINE bool isBoundaryEdge() const { return !valid_left || !valid_right; }
};

/**
 * @brief A function to sample a shape and it's edges uniformly. Because of this each point has the same pdf
 */
struct ShapeSamplerVPtrTable
{
    uint32_t sampleShapeCallIndex;
    uint32_t sampleEdgeCallIndex;
    uint32_t evalShapePdfCallIndex;
    uint32_t evalEdgePdfCallIndex;

#ifdef __CUDACC__
    __device__ ShapeSampleResult sampleShape(PCG32& rng) const
    {
        return optixDirectCall<ShapeSampleResult, PCG32&>(sampleShapeCallIndex, rng);
    }

    __device__ EdgeSampleResult sampleEdge(PCG32& rng) const
    {
        return optixDirectCall<EdgeSampleResult, PCG32&>(sampleEdgeCallIndex, rng);
    }

    // Pdf in area domain. Because of uniform sampling, this is the same for all points on the shape
    __device__ float evalShapePdf() const { return optixDirectCall<float>(evalShapePdfCallIndex); }

    // Pdf in area domain. Because of uniform sampling, this is the same for all edge points
    __device__ float evalEdgePdf() const { return optixDirectCall<float>(evalEdgePdfCallIndex); }
#endif
};
}    // namespace atcg