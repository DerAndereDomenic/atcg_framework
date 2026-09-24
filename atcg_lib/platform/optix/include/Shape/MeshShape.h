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
class ATCG_API MeshShape : public Shape, public std::enable_shared_from_this<MeshShape>
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
     * @brief Create a shape sampler for this shape
     * @param transform The transform of the shape sampler
     * @return A shape sampler for this shape
     */
    virtual atcg::ref_ptr<ShapeSampler> createSampler(const glm::mat4& transform) override;

    /**
     * @brief Get the shape type
     * @return The shape type
     */
    virtual std::string getShapeType() const override { return "MeshShape"; }

    ATCG_INLINE torch::Tensor getPositions() const { return _positions; }
    ATCG_INLINE torch::Tensor getNormals() const { return _normals; }
    ATCG_INLINE torch::Tensor getColors() const { return _colors; }
    ATCG_INLINE torch::Tensor getUVs() const { return _uvs; }
    ATCG_INLINE torch::Tensor get3DFaces() const { return _faces_3d; }
    ATCG_INLINE torch::Tensor getUVFaces() const { return _faces_uv; }
    ATCG_INLINE torch::Tensor getNormalFaces() const { return _faces_normals; }
    ATCG_INLINE torch::Tensor getColorFaces() const { return _faces_color; }
    ATCG_INLINE torch::Tensor getEdges() const { return _edges; }
    ATCG_INLINE torch::Tensor getEdgeFaces() const { return _edge_faces; }
    ATCG_INLINE atcg::dref_ptr<MeshShapeData> getMeshShapeData() const { return _data; }

private:
    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _colors;
    torch::Tensor _uvs;
    torch::Tensor _faces_3d;
    torch::Tensor _faces_uv;
    torch::Tensor _faces_normals;
    torch::Tensor _faces_color;
    torch::Tensor _edges;
    torch::Tensor _edge_faces;

    atcg::dref_ptr<MeshShapeData> _data;
};
}    // namespace atcg