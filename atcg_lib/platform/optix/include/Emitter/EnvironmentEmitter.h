#pragma once

#include <Renderer/Texture.h>
#include <Emitter/Emitter.h>
#include <Emitter/EnvironmentEmitterData.cuh>
#include <Core/PipelineInitializer.h>

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

    ATCG_INLINE atcg::dref_ptr<EnvironmentEmitterData> getDataBuffer() const { return _environment_emitter_data; }

private:
    atcg::ref_ptr<Texture2D> _environment_texture;

    atcg::dref_ptr<EnvironmentEmitterData> _environment_emitter_data;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(EnvironmentEmitter);
}    // namespace atcg