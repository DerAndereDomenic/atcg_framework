#pragma once

#include <Shape/ShapeSampler.h>
#include <Shape/MeshSamplerData.cuh>

namespace atcg
{
class ATCG_API MeshShapeSampler : public ShapeSampler
{
public:
    MeshShapeSampler(const atcg::Dictionary& dict);

    virtual ~MeshShapeSampler();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {};

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    torch::Tensor _mesh_cdf;

    atcg::dref_ptr<MeshSamplerData> _data;
};
}    // namespace atcg