#pragma once

#include <Core/Platform.h>
#include <DataStructure/Dictionary.h>
#include <Core/OptixComponent.h>
#include <BSDF/BSDFVPtrTable.cuh>

namespace atcg
{
/**
 * @brief A class to model a BSDF
 */
class BSDF : public OptixComponent
{
public:
    /**
     * @brief Constructor
     */
    BSDF() = default;

    /**
     * @brief Construct a BSDF with arbitrary parameters
     *
     * @param dict Dictionary holding the parameters
     */
    BSDF(const atcg::Dictionary& dict) {}

    /**
     * @brief Destructor
     */
    virtual ~BSDF() {}

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    /**
     * @brief Get the VPtrTable
     *
     * @return The VPtrTable
     */
    ATCG_INLINE const BSDFVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief Get the bsdf flags
     *
     * @return The flags
     */
    ATCG_INLINE const BSDFComponentType& flags() const { return _flags; }

protected:
    atcg::dref_ptr<BSDFVPtrTable> _vptr_table;
    BSDFComponentType _flags;
};
}    // namespace atcg