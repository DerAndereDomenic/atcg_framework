#pragma once

#include <Renderer/Emitter.cuh>
#include <Renderer/ShaderBindingTable.h>
#include <Renderer/RaytracingPipeline.h>
#include <Renderer/Material.h>
#include <DataStructure/Graph.h>

namespace atcg
{
class Emitter
{
public:
    Emitter() = default;

    virtual ~Emitter() = default;

    inline const EmitterVPtrTable* getEmitterVPtrTable() const { return _vptr_table.get(); }

    virtual void initializeEmitter(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                   const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
};

class MeshEmitter : public Emitter
{
public:
    MeshEmitter(const atcg::ref_ptr<Graph>& graph, const Material& material);

    ~MeshEmitter();

    virtual void initializeEmitter(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
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

class EnvironmentEmitter : public Emitter
{
public:
    EnvironmentEmitter(const atcg::ref_ptr<atcg::Texture2D>& environment_texture);

    ~EnvironmentEmitter();

    virtual void initializeEmitter(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                   const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    cudaArray_t _environment_texture;

    atcg::dref_ptr<EnvironmentEmitterData> _environment_emitter_data;
};
}    // namespace atcg