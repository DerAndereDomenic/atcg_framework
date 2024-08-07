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
/**
 * @brief A class to model an optix component
 */
class OptixComponent
{
public:
    /**
     * @brief Initialize the optix pipeline.
     * Each Optix component has to initialize its part of the raytracing pipeline by defining appropriate entry points
     * and sbt entries.
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;
};

/**
 * @brief Optix interface for BSDFs
 */
class OptixBSDF : public OptixComponent, public BSDF
{
public:
    /**
     * @brief Destructor
     */
    virtual ~OptixBSDF() {}

    /**
     * @brief Get the vptr table to call bsdf functions on the GPU
     *
     * @return The vptr table
     */
    inline const BSDFVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<BSDFVPtrTable> _vptr_table;
};

/**
 * @brief Optix interface for Emitters
 */
class OptixEmitter : public OptixComponent, public Emitter
{
public:
    /**
     * @brief Destructor
     */
    virtual ~OptixEmitter() {}

    /**
     * @brief Get the vptr table to call Emitters functions on the GPU
     *
     * @return The vptr table
     */
    inline const EmitterVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
};

}    // namespace atcg