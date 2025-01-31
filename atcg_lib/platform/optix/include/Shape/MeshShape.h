#pragma once

#include <Shape/Shape.h>
#include <DataStructure/Graph.h>
#include <DataStructure/TorchUtils.h>

namespace atcg
{
class MeshShape : public Shape
{
public:
    MeshShape(const atcg::ref_ptr<Graph>& mesh);

    virtual ~MeshShape();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    virtual void prepareAccelerationStructure(OptixDeviceContext context) override;

private:
    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _uvs;
    torch::Tensor _faces;
};
}    // namespace atcg