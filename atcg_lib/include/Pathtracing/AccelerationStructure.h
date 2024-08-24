#pragma once

#include <DataStructure/Graph.h>
#include <Scene/Scene.h>
#include <torch/types.h>
#include <nanort.h>
#include <Pathtracing/PathtracingPlatform.h>
#include <Pathtracing/ShaderBindingTable.h>
#include <Pathtracing/RaytracingPipeline.h>

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

class BVHAccelerationStructure : public AccelerationStructure
{
public:
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

protected:
    nanort::BVHAccel<float> _bvh;
};

/**
 * @brief A BVH acceleration structure for CPU-based ray triangle intersection
 */
class GASAccelerationStructure : public BVHAccelerationStructure
{
public:
    /**
     * @brief Default Constructor
     */
    GASAccelerationStructure() = default;

    /**
     * @brief Constructor
     *
     * @param graph The geometry to build the BVH
     */
    GASAccelerationStructure(const atcg::ref_ptr<Graph>& graph);

    /**
     * @brief Destructor
     */
    ~GASAccelerationStructure();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt);

    inline ATCGProgramGroup getHitGroup() const { return _hit_group; }

private:
    ATCGProgramGroup _hit_group;
};

class IASAccelerationStructure : public BVHAccelerationStructure
{
public:
    IASAccelerationStructure() = default;

    IASAccelerationStructure(const atcg::ref_ptr<Scene>& scene);

    ~IASAccelerationStructure();

    inline torch::Tensor getMeshIDs() const { return _mesh_idx; }

    inline atcg::ref_ptr<Scene> getScene() const { return _scene; }

    inline const std::vector<glm::mat4>& getTransforms() const { return _transforms; }

    inline const std::vector<uint32_t>& getOffsets() const { return _offsets; }

private:
    torch::Tensor _mesh_idx;
    std::vector<glm::mat4> _transforms;
    std::vector<uint32_t> _offsets;
    atcg::ref_ptr<Scene> _scene;
};

}    // namespace atcg