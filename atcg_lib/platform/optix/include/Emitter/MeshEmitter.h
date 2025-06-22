#pragma once

#include <Renderer/Texture.h>
#include <Emitter/Emitter.h>
#include <Emitter/MeshEmitterData.cuh>
#include <DataStructure/Dictionary.h>

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
    ~MeshEmitter();

    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _emissive_texture;

    torch::Tensor _mesh_cdf;

    atcg::dref_ptr<MeshEmitterData> _mesh_emitter_data;
};
}    // namespace atcg