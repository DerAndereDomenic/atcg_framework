#pragma once

#include <Integrator/Integrator.h>
#include <OpenMesh/OpenMesh.h>
#include <Shape/MeshShape.h>
#include <Shape/IAS.h>
#include "RadiosityParams.h"

class RadiosityRayGenerator : public atcg::Integrator
{
public:
    RadiosityRayGenerator(const atcg::ref_ptr<atcg::RaytracingContext>& context) : atcg::Integrator(context) {}

    virtual ~RadiosityRayGenerator() {}

    virtual void initializePipeline(const atcg::ref_ptr<atcg::RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<atcg::ShaderBindingTable>& sbt) override;

    virtual void generateRays(const atcg::ref_ptr<atcg::PerspectiveCamera>& camera,
                              const std::vector<torch::Tensor>& output) override;

    virtual void reset() override;

    void setMesh(const atcg::ref_ptr<atcg::TriMesh>& mesh);

private:
    atcg::ref_ptr<atcg::TriMesh> _mesh;
    atcg::ref_ptr<atcg::MeshShape> _shape;

    uint32_t _raygen_index;
    uint32_t _occlusion_miss_index;

    atcg::dref_ptr<RadiosityParams> _launch_params;

    std::vector<atcg::ref_ptr<atcg::ShapeInstance>> _shapes;
    atcg::ref_ptr<atcg::InstanceAccelerationStructure> _ias;
};