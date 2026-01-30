#include "RadiosityRayGenerator.h"

#include <Core/Common.h>
#include <DataStructure/Graph.h>

#include <optix_stubs.h>

void RadiosityRayGenerator::initializePipeline()
{
    _pipeline->addTrianglesHitGroupShader("MeshShape", 0, {"./bin/MeshShape_ptx.ptx", "__closesthit__mesh"}, {});

    auto graph = atcg::Graph::createTriangleMesh(_mesh);
    atcg::Dictionary dict;
    dict.setValue("mesh", graph);
    _shape = atcg::make_ref<atcg::MeshShape>(dict);

    _shape->initializePipeline(_pipeline, _sbt);
    _shape->prepareAccelerationStructure(_context);

    atcg::Dictionary shape_data;
    shape_data.setValue("shape", _shape);
    auto shape_instance = atcg::make_ref<atcg::ShapeInstance>(shape_data);
    shape_instance->initializePipeline(_pipeline, _sbt);
    _shapes.push_back(shape_instance);

    const std::string ptx_raygen_filename = "./bin/RadiosityRayGenerator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = _pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup occl_prog_group     = _pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = _sbt->addRaygenEntry(raygen_prog_group);
    _occlusion_miss_index = _sbt->addMissEntry(occl_prog_group);

    _ias = atcg::make_ref<atcg::InstanceAccelerationStructure>(_context, _shapes, _pipeline->numRays());

    _pipeline->createPipeline();
    _sbt->createSBT();
}

void RadiosityRayGenerator::generateRays(atcg::Dictionary& dict)
{
    auto output           = dict.getValue<torch::Tensor>("output");
    uint32_t n_primitives = output.size(0);

    RadiosityParams params;

    params.handle                 = _ias->getTraversableHandle();
    params.occlusion_trace_params = _pipeline->getRay(0, _occlusion_miss_index, true);

    params.form_factors = (float*)output.data_ptr();
    params.shape        = std::static_pointer_cast<atcg::MeshShape>(_shape)->getMeshShapeData().get();
    params.n_faces      = n_primitives;

    _launch_params.upload(&params);

    OPTIX_CHECK(optixLaunch(_pipeline->getPipeline(),
                            nullptr,
                            (CUdeviceptr)_launch_params.get(),
                            sizeof(RadiosityParams),
                            _sbt->getSBT(_raygen_index),
                            n_primitives,
                            n_primitives,
                            1));    // depth

    CUDA_SAFE_CALL(cudaStreamSynchronize(nullptr));
}

void RadiosityRayGenerator::reset() {}