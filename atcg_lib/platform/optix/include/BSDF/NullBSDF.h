#pragma once

#include <BSDF/BSDF.h>
#include <Renderer/Material.h>

namespace atcg
{
/**
 * @brief A PBR BSDF
 */
class NullBSDF : public BSDF
{
public:
    /**
     * @brief Construct a Null BSDF
     *
     * @param dict Dictionary holding the parameters (unused)
     */
    NullBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    ~NullBSDF();

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

private:
};
}    // namespace atcg