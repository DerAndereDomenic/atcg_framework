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
     * -"mesh": atcg::ref_ptr<Graph>
     *
     * @param dict The parameters
     */
    MeshShape(const Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~MeshShape();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    /**
     * @brief Prepare the geometry acceleration structor
     *
     * @param context The raytracing context
     */
    virtual void prepareAccelerationStructure(const atcg::ref_ptr<RaytracingContext>& context) override;

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    /**
     * @brief Get the shape type
     * @return The shape type
     */
    virtual std::string getShapeType() const override { return "MeshShape"; }

    ATCG_INLINE torch::Tensor getPositions() const { return _positions; }
    ATCG_INLINE torch::Tensor getNormals() const { return _normals; }
    ATCG_INLINE torch::Tensor getColors() const { return _colors; }
    ATCG_INLINE torch::Tensor getUVs() const { return _uvs; }
    ATCG_INLINE torch::Tensor getFaces() const { return _faces; }
    ATCG_INLINE torch::Tensor getEdges() const { return _edges; }
    ATCG_INLINE atcg::dref_ptr<MeshShapeData> getMeshShapeData() const { return _data; }

private:
    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _colors;
    torch::Tensor _uvs;
    torch::Tensor _faces;
    torch::Tensor _edges;

    atcg::dref_ptr<MeshShapeData> _data;
};
}    // namespace atcg