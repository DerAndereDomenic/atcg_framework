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
    float pdf_A;
};

struct EdgeSampleResult
{
    glm::vec3 position;
    glm::vec3 normal_left;
    glm::vec3 normal_right;
    float pdf_A;
};

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
#endif
};
}    // namespace atcg