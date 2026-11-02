#include "PerfectDielectricBSDF.h"

#include <Renderer/Texture.h>
#include "PerfectDielectric.h"
#include <Renderer/Material.h>

#include <Core/Common.h>

namespace atcg
{

PerfectDielectricBSDF::PerfectDielectricBSDF(const Dictionary& dict)
{
    auto material = std::dynamic_pointer_cast<atcg::PerfectDielectric>(dict.getValue<atcg::ref_ptr<Material>>("materia"
                                                                                                              "l"));
    auto ior_texture = material->getIORTexture()->getData(atcg::GPU);
    _ior_texture     = ior_texture;

    PerfectDielectricData data;

    data.ior_texture =
        TextureSampler<float>((std::byte*)ior_texture.data_ptr(), material->getIORTexture()->getSpecification());

    _flags = BSDFComponentType::IdealReflection | BSDFComponentType::IdealTransmission;

    _bsdf_data_buffer.upload(&data);
}

PerfectDielectricBSDF::~PerfectDielectricBSDF() {}

void PerfectDielectricBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/PerfectDielectricBSDF_ptx.ptx";
    auto sample_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_perfectdielectricbsdf"});
    auto eval_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_perfectdielectricbsdf"});
    auto eval_pdf_prog_group =
        pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__evalpdf_perfectdielectricbsdf"});

    uint32_t sample_idx   = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx     = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_pdf_idx = sbt->addCallableEntry(eval_pdf_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex  = sample_idx;
    table.evalCallIndex    = eval_idx;
    table.evalPDFCallIndex = eval_pdf_idx;
    table.flags            = _flags;

    _vptr_table.upload(&table);

    markInitialized();
}

void PerfectDielectricBSDF::registerBSDF(BSDFRegistry::Registry* registry)
{
    ATCG_REGISTER_BSDF(registry, "PerfectDielectric", PerfectDielectricBSDF);
}
}    // namespace atcg