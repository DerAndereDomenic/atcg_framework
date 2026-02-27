#pragma once

#include <Renderer/Texture.h>
#include <Emitter/Emitter.h>
#include <Emitter/EnvironmentEmitterData.cuh>

namespace atcg
{
/**
 * @brief An environment emitter
 */
class EnvironmentEmitter : public Emitter
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

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    /**
     * @brief Initialize the pipeline
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    torch::Tensor _environment_texture;

    atcg::dref_ptr<EnvironmentEmitterData> _environment_emitter_data;
};

}    // namespace atcg