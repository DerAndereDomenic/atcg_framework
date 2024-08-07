#pragma once

#include <torch/types.h>
#include <DataStructure/Graph.h>
#include <nanort.h>

namespace atcg
{
/**
 * @brief A class to wrap around a acceleration structure.
 * Backend dependent
 */
class AccelerationStructure
{
public:
    /**
     * @brief Constructor
     */
    AccelerationStructure() = default;

    /**
     * @brief Destructor
     */
    virtual ~AccelerationStructure() {}

    /**
     * @brief Get the positions
     *
     * @return The positions
     */
    inline torch::Tensor getPositions() const { return _positions; }

    /**
     * @brief Get the normals
     *
     * @return The normals
     */
    inline torch::Tensor getNormals() const { return _normals; }

    /**
     * @brief Get the uvs
     *
     * @return The uvs
     */
    inline torch::Tensor getUVs() const { return _uvs; }

    /**
     * @brief Get the faces
     *
     * @return The faces
     */
    inline torch::Tensor getFaces() const { return _faces; }

protected:
    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _uvs;
    torch::Tensor _faces;
};

/**
 * @brief A BVH acceleration structure for CPU-based ray triangle intersection
 */
class BVHAccelerationStructure : public AccelerationStructure
{
public:
    /**
     * @brief Default Constructor
     */
    BVHAccelerationStructure() = default;

    /**
     * @brief Constructor
     *
     * @param graph The geometry to build the BVH
     */
    BVHAccelerationStructure(const atcg::ref_ptr<Graph>& graph);

    /**
     * @brief Destructor
     */
    ~BVHAccelerationStructure();

    /**
     * @brief Get the acceleration structure handle
     *
     * @return The handle
     */
    inline const nanort::BVHAccel<float>& getBVH() const { return _bvh; }

    /**
     * @brief Set the acceleration structure handle
     *
     * @param bvh The handle
     */
    inline void setBVH(const nanort::BVHAccel<float>& bvh) { _bvh = bvh; }

private:
    nanort::BVHAccel<float> _bvh;
};

}    // namespace atcg