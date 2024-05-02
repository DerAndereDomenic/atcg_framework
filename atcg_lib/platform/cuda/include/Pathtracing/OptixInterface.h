#pragma once

#include <Core/Memory.h>

#include <Pathtracing/ShaderBindingTable.h>
#include <Pathtracing/RaytracingPipeline.h>

#include <Pathtracing/BSDF.h>
#include <Pathtracing/BSDFModels.cuh>

#include <Pathtracing/Emitter.h>
#include <Pathtracing/EmitterModels.cuh>

#include <Pathtracing/RaytracingShader.h>

#include <torch/types.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
class Scene;
class OptixComponent
{
public:
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;
};

class OptixBSDF : public OptixComponent, public BSDF
{
public:
    inline const BSDFVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<BSDFVPtrTable> _vptr_table;
};

class OptixEmitter : public OptixComponent, public Emitter
{
public:
    inline const EmitterVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
};

class OptixRaytracingShader : public OptixComponent, public RaytracingShader
{
public:
    OptixRaytracingShader(OptixDeviceContext context,
                          const atcg::ref_ptr<Scene>& scene,
                          const atcg::ref_ptr<PerspectiveCamera>& camera)
        : _context(context),
          _scene(scene)
    {
    }

    virtual ~OptixRaytracingShader() {}


    virtual void reset() = 0;

    virtual void setCamera(const atcg::ref_ptr<PerspectiveCamera>& camera) = 0;

    virtual void generateRays(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                              const atcg::ref_ptr<ShaderBindingTable>& sbt,
                              torch::Tensor& output) = 0;

protected:
    OptixDeviceContext _context;
    atcg::ref_ptr<Scene> _scene;
};

}    // namespace atcg