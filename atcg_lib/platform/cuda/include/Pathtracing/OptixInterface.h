#pragma once

#include <Core/Memory.h>

#include <Pathtracing/ShaderBindingTable.h>
#include <Pathtracing/RaytracingPipeline.h>

#include <Pathtracing/BSDF.h>
#include <Pathtracing/BSDF/BSDFModels.cuh>

#include <Pathtracing/Emitter.h>
#include <Pathtracing/Emitter/EmitterModels.cuh>

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
    virtual ~OptixBSDF() {}

    inline const BSDFVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<BSDFVPtrTable> _vptr_table;
};

class OptixEmitter : public OptixComponent, public Emitter
{
public:
    virtual ~OptixEmitter() {}

    inline const EmitterVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
};

}    // namespace atcg