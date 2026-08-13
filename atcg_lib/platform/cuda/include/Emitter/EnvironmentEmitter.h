#pragma once

#include <Renderer/Texture.h>
#include <Emitter/Emitter.h>
#include <Emitter/EnvironmentEmitterData.cuh>

namespace atcg
{
/**
 * @brief An environment emitter
 */
class ATCG_API EnvironmentEmitter : public Emitter
{
public:
    /**
     * @brief Create an emitter
     * - "environment_texture": atcg::ref_ptr<atcg::Texture2D>, an equirectangular environment map
     *
     * @param dict The parameters
     */
    EnvironmentEmitter(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~EnvironmentEmitter();

    // TODO
    virtual void updateData() override {}

    /**
     * @brief Initialize the pipeline
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::ref_ptr<Texture2D> _environment_texture;

    atcg::dref_ptr<EnvironmentEmitterData> _environment_emitter_data;

    torch::Tensor _col_pdfs;
    torch::Tensor _col_cdfs;

    torch::Tensor _row_pdf;
    torch::Tensor _row_cdf;
};

}    // namespace atcg