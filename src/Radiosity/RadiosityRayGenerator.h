#pragma once

#include <Integrator/Integrator.h>
#include <OpenMesh/OpenMesh.h>
#include <Shape/MeshShape.h>
#include <Shape/IAS.h>
#include "RadiosityParams.h"

class RadiosityRayGenerator : public atcg::Integrator
{
public:
    RadiosityRayGenerator(const atcg::ref_ptr<atcg::RaytracingContext>& context, const atcg::Dictionary& dict)
        : atcg::Integrator(context, dict)
    {
        _mesh = dict.getValue<atcg::ref_ptr<atcg::TriMesh>>("mesh");
        initializePipeline();
    }

    virtual ~RadiosityRayGenerator() {}

    virtual void generateRays(atcg::Dictionary& dict) override;

    virtual void reset() override;

    virtual void onImGuiRender() override {}

private:
    void initializePipeline();
    atcg::ref_ptr<atcg::TriMesh> _mesh;
    atcg::ref_ptr<atcg::Shape> _shape;

    uint32_t _raygen_index;
    uint32_t _occlusion_miss_index;

    atcg::dref_ptr<RadiosityParams> _launch_params;

    std::vector<atcg::ref_ptr<atcg::ShapeInstance>> _shapes;
    atcg::ref_ptr<atcg::InstanceAccelerationStructure> _ias;
};