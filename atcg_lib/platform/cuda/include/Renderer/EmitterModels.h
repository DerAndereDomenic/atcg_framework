#pragma once

#include <Renderer/OptixInterface.h>
#include <Renderer/Material.h>
#include <DataStructure/Graph.h>

namespace atcg
{

class MeshEmitter : public OptixEmitter
{
public:
    MeshEmitter(const atcg::ref_ptr<Graph>& graph, const glm::mat4& transform, const Material& material);

    ~MeshEmitter();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _emissive_texture;

    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _uvs;
    torch::Tensor _faces;

    torch::Tensor _mesh_cdf;

    atcg::dref_ptr<MeshEmitterData> _mesh_emitter_data;
};

class EnvironmentEmitter : public OptixEmitter
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