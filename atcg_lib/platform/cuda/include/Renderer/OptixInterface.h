#pragma once

#include <Core/Memory.h>

#include <Renderer/ShaderBindingTable.h>
#include <Renderer/RaytracingPipeline.h>

#include <Renderer/BSDF.h>
#include <Renderer/BSDFModels.cuh>

#include <Renderer/Emitter.h>
#include <Renderer/EmitterModels.cuh>

namespace atcg
{
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

}    // namespace atcg