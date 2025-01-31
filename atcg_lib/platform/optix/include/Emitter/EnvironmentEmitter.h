#pragma once

#include <Renderer/Texture.h>
#include <Emitter/Emitter.h>
#include <Emitter/EnvironmentEmitterData.cuh>

namespace atcg
{
class EnvironmentEmitter : public Emitter
{
public:
    EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& environment_texture);

    ~EnvironmentEmitter();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _environment_texture;

    atcg::dref_ptr<EnvironmentEmitterData> _environment_emitter_data;
};
}    // namespace atcg