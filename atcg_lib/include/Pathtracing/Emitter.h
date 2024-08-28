#pragma once

#include <Core/glm.h>
#include <Pathtracing/EmitterModels.cuh>
#include <Pathtracing/ShaderBindingTable.h>
#include <Pathtracing/RaytracingPipeline.h>

namespace atcg
{
/**
 * @brief Class to model an emitter. This class is only used for high level storage. The real interface of an emitter is
 * backend dependent.
 */
class Emitter
{
public:
    /**
     * @brief Constructor
     */
    Emitter() = default;

    /**
     * @brief Destructor
     */
    virtual ~Emitter() {}

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    const EmitterVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
};
}    // namespace atcg