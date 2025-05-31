#include "RadiosityRayGenerator.h"

#include <Core/Common.h>
#include <DataStructure/Graph.h>

void RadiosityRayGenerator::initializePipeline(const atcg::ref_ptr<atcg::RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<atcg::ShaderBindingTable>& sbt)
{
    auto graph = atcg::Graph::createTriangleMesh(_mesh);
    _shape     = atcg::make_ref<atcg::MeshShape>(graph);

    _shape->initializePipeline(pipeline, sbt);
    _shape->prepareAccelerationStructure(_context);

    atcg::Dictionary shape_data;
    shape_data.setValue("shape", _shape);
    auto shape_instance = atcg::make_ref<atcg::ShapeInstance>(shape_data);
    shape_instance->initializePipeline(pipeline, sbt);
    _shapes.push_back(shape_instance);

    const std::string ptx_raygen_filename = "./bin/RadiosityRayGenerator_ptx.ptx";
    OptixProgramGroup raygen_prog_group   = pipeline->addRaygenShader({ptx_raygen_filename, "__raygen__rg"});
    OptixProgramGroup occl_prog_group     = pipeline->addMissShader({ptx_raygen_filename, "__miss__occlusion"});

    _raygen_index         = sbt->addRaygenEntry(raygen_prog_group);
    _occlusion_miss_index = sbt->addMissEntry(occl_prog_group);

    _ias = atcg::make_ref<atcg::InstanceAccelerationStructure>(_context, _shapes);

    _pipeline = pipeline;
    _sbt      = sbt;
}

void RadiosityRayGenerator::generateRays(atcg::Dictionary& dict)
{
    auto output           = dict.getValue<torch::Tensor>("output");
    uint32_t n_primitives = output.size(0);

    RadiosityParams params;

    params.handle                           = _ias->getTraversableHandle();
    params.occlusion_trace_params.rayFlags  = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    params.occlusion_trace_params.SBToffset = 0;
    params.occlusion_trace_params.SBTstride = 1;
    params.occlusion_trace_params.missSBTIndex = _occlusion_miss_index;

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

void RadiosityRayGenerator::setMesh(const atcg::ref_ptr<atcg::TriMesh>& mesh)
{
    _mesh = mesh;
}