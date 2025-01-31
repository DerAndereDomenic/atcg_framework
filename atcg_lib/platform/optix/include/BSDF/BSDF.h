#pragma once

#include <Core/Platform.h>
#include <Core/OptixComponent.h>
#include <BSDF/BSDFVPtrTable.cuh>

namespace atcg
{
class BSDF : public OptixComponent
{
public:
    virtual ~BSDF() {}

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    ATCG_INLINE const BSDFVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    ATCG_INLINE const BSDFComponentType& flags() const { return _flags; }

protected:
    atcg::dref_ptr<BSDFVPtrTable> _vptr_table;
    BSDFComponentType _flags;
};
}    // namespace atcg