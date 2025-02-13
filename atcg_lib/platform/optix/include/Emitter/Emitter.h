#pragma once

#include <Core/Platform.h>
#include <Core/OptixComponent.h>
#include <Emitter/EmitterVPtrTable.cuh>

namespace atcg
{
class Emitter : public OptixComponent
{
public:
    virtual ~Emitter() {}

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    inline const EmitterVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
    EmitterFlags _flags;
};
}    // namespace atcg