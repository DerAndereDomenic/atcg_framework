#include <BSDF/DielectricBSDF.h>

#include <Renderer/Texture.h>

#include <Core/Common.h>

namespace atcg
{

DielectricBSDF::DielectricBSDF(const Dictionary& dict)
{
    auto transmittance_texture = dict.getValue<atcg::ref_ptr<Texture2D>>("transmittance");
    auto reflectance_texture   = dict.getValue<atcg::ref_ptr<Texture2D>>("reflectance");
    float ior                  = dict.getValue<float>("ior");

    auto transmittance = transmittance_texture->getData(atcg::GPU);
    auto reflectance   = reflectance_texture->getData(atcg::GPU);

    DielectricBSDFData data;

    atcg::convertToTextureObject(reflectance, _reflectance_texture, data.reflectance_texture);
    atcg::convertToTextureObject(transmittance, _transmittance_texture, data.transmittance_texture);
    data.ior = ior;

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealReflection;

    _bsdf_data_buffer.upload(&data);
}

DielectricBSDF::~DielectricBSDF()
{
    DielectricBSDFData data;

    _bsdf_data_buffer.download(&data);

    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.reflectance_texture));
    CUDA_SAFE_CALL(cudaDestroyTextureObject(data.transmittance_texture));

    CUDA_SAFE_CALL(cudaFreeArray(_reflectance_texture));
    CUDA_SAFE_CALL(cudaFreeArray(_transmittance_texture));
}

void DielectricBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                        const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/DielectricBSDF_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_dielectricbsdf"});
    auto eval_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_dielectricbsdf"});
    uint32_t sample_idx  = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx    = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _vptr_table.upload(&table);
}
}    // namespace atcg