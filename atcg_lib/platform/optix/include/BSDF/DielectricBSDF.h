#pragma once

#include <BSDF/BSDF.h>
#include <BSDF/DielectricBSDFData.cuh>

namespace atcg
{
class DielectricBSDF : public BSDF
{
public:
    /**
     * @brief Construct a BSDF with arbitrary parameters.
     * Input parameters:
     * - "material": atcg::Material
     *
     * @param dict Dictionary holding the parameters
     */
    DielectricBSDF(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    ~DielectricBSDF();

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
    cudaArray_t _reflectance_texture;
    cudaArray_t _transmittance_texture;

    atcg::dref_ptr<DielectricBSDFData> _bsdf_data_buffer;
};
}    // namespace atcg