#pragma once

#include <Renderer/Texture.h>
#include <Emitter/Emitter.h>
#include <Emitter/MeshEmitterData.cuh>
#include <DataStructure/Dictionary.h>
#include <Core/PipelineInitializer.h>

namespace atcg
{
/**
 * @brief A mesh emitter
 */
class MeshEmitter : public Emitter
{
public:
    /**
     * @brief Constructor from a dictionary.
     * The dictionary expects
     * - shape: atcg::ref_ptr<MeshShape>
     * - transform: glm::mat4
     * - texture_emissive: atcg::ref_ptr<Texture2D>
     * - emission_scaling: float
     *
     * @param dict The shape data
     */
    MeshEmitter(const Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~MeshEmitter();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    ATCG_INLINE atcg::dref_ptr<MeshEmitterData> getDataBuffer() const { return _mesh_emitter_data; }

private:
    atcg::ref_ptr<Texture2D> _emissive_texture;

    torch::Tensor _mesh_cdf;

    atcg::dref_ptr<MeshEmitterData> _mesh_emitter_data;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(MeshEmitter);
}    // namespace atcg