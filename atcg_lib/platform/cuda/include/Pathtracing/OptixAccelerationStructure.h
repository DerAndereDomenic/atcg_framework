#pragma once

#include <Pathtracing/AccelerationStructure.h>
#include <Pathtracing/OptixInterface.h>
#include <optix.h>

#include <Scene/Scene.h>

namespace atcg
{

class OptixAccelerationStructure : public AccelerationStructure, public OptixComponent
{
public:
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override = 0;

    OptixTraversableHandle getTraversableHandle() const { return _handle; }

    OptixProgramGroup getHitGroup() const { return _hit_group; }

protected:
    atcg::DeviceBuffer<uint8_t> _ast_buffer;
    OptixTraversableHandle _handle;
    OptixProgramGroup _hit_group;
};

class MeshAccelerationStructure : public OptixAccelerationStructure
{
public:
    MeshAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Graph>& graph);

    virtual ~MeshAccelerationStructure();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
};

class IASAccelerationStructure : public OptixAccelerationStructure
{
public:
    IASAccelerationStructure(OptixDeviceContext context, const atcg::ref_ptr<Scene>& scene);

    ~IASAccelerationStructure();

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::ref_ptr<Scene> _scene;
};
}    // namespace atcg