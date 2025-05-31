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
     * @brief Constructor
     * 
     * @param environment_texture The equirectangular environment map
     */
    EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& environment_texture);

    /**
     * @brief Destructor
     */
    ~EnvironmentEmitter();

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
    cudaArray_t _environment_texture;

    atcg::dref_ptr<EnvironmentEmitterData> _environment_emitter_data;
};
}    // namespace atcg