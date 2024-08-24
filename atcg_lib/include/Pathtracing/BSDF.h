#pragma once

#include <Core/glm.h>
#include <Core/Memory.h>
#include <Core/CUDA.h>
#include <Pathtracing/BSDFModels.cuh>
#include <Pathtracing/ShaderBindingTable.h>
#include <Pathtracing/RaytracingPipeline.h>

namespace atcg
{
/**
 * @brief A class to model a BSDF. This is only a high level interface. The true implementation is backend dependent.
 */
class BSDF
{
public:
    /**
     * @brief Constructor
     */
    BSDF() = default;

    /**
     * @brief Destructor
     */
    virtual ~BSDF() {}

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    inline const BSDFVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief Get the bsdf flags
     *
     * @return The flags
     */
    ATCG_INLINE const BSDFComponentType& flags() const { return _flags; }

protected:
    BSDFComponentType _flags;
    atcg::ref_ptr<BSDFVPtrTable> _vptr_table;
};

}    // namespace atcg