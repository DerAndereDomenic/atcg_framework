#pragma once

#include <Renderer/Material.h>
#include <DataStructure/Graph.h>
#include <Pathtracing/Emitter.h>
#include <Pathtracing/EmitterModels.cuh>

namespace atcg
{

/**
 * @brief A mesh emitter
 */
class MeshEmitter : public Emitter
{
public:
    /**
     * @brief Create a mesh emitter
     *
     * @param positions The mesh positions
     * @param uvs The uv coordinates
     * @param faces The faces
     * @param transform The mesh transform matrix
     * @param material The emitter material
     */
    MeshEmitter(const torch::Tensor& positions,
                const torch::Tensor& uvs,
                const torch::Tensor& faces,
                const glm::mat4& transform,
                const Material& material);

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
    atcg::ref_ptr<MeshEmitterData> _mesh_emitter_data;
};

/**
 * @brief Class to model an environment emitter
 */
class EnvironmentEmitter : public Emitter
{
public:
    /**
     * @brief Create an environment emitter
     *
     * @param environment_texture The environment texture
     */
    EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& environment_texture);

    /**
     * @brief Destructor
     */
    ~EnvironmentEmitter();

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
    atcg::ref_ptr<EnvironmentEmitterData> _environment_emitter_data;
};
}    // namespace atcg