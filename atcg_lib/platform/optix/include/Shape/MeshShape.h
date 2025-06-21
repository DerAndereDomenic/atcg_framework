#pragma once

#include <Shape/Shape.h>
#include <Shape/MeshShapeData.cuh>
#include <DataStructure/Graph.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
/**
 * @brief A class to model a triangle mesh
 */
class MeshShape : public Shape
{
public:
    /**
     * @brief Constructor
     *
     * @param mesh The mesh to render
     */
    MeshShape(const atcg::ref_ptr<Graph>& mesh);

    /**
     * @brief Destructor
     */
    virtual ~MeshShape();

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    /**
     * @brief Prepare the geometry acceleration structor
     *
     * @param context The raytracing context
     */
    virtual void prepareAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context) override;

    /**
     * @brief Get the Mesh Shape data
     *
     * @return The data
     */
    ATCG_INLINE atcg::dref_ptr<MeshShapeData> getMeshShapeData() const { return _data; }

private:
    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _colors;
    torch::Tensor _uvs;
    torch::Tensor _faces;

    atcg::dref_ptr<MeshShapeData> _data;
};
}    // namespace atcg