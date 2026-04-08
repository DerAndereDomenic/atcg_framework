#include <BSDF/NeuralBSDF.h>

#include <DataStructure/TorchUtils.h>

#include <Core/Common.h>

namespace atcg
{
NeuralBSDF::NeuralBSDF(const atcg::Dictionary& dict) : BSDF(dict)
{
    auto context = dict.getValue<atcg::ref_ptr<RaytracingContext>>("context");
    // _weights =
    //     torch::randn({8 * 64 + 64 * 64 + 64 * 8}, torch::TensorOptions {}.device(atcg::GPU).dtype(torch::kFloat16));
    // _bias = torch::randn({64 + 64 + 8}, torch::TensorOptions {}.device(atcg::GPU).dtype(torch::kFloat16));

    // Load binary data for weights and bias
    std::ifstream weights_file("C:/Users/zingsheim/Documents/Repositories/PythonTest/weights.bin",
                               std::ios::in | std::ios::binary);
    std::vector<uint8_t> weight_buffer_char(std::istreambuf_iterator<char>(weights_file), {});
    weights_file.close();

    std::ifstream bias_file("C:/Users/zingsheim/Documents/Repositories/PythonTest/biases.bin",
                            std::ios::in | std::ios::binary);
    std::vector<uint8_t> bias_buffer_char(std::istreambuf_iterator<char>(bias_file), {});
    bias_file.close();

    _weights = torch::from_blob(weight_buffer_char.data(),
                                {(int)(weight_buffer_char.size() / sizeof(half))},
                                torch::TensorOptions {}.device(atcg::CPU).dtype(torch::kFloat16))
                   .cuda();
    _bias    = torch::from_blob(bias_buffer_char.data(),
                                {(int)(bias_buffer_char.size() / sizeof(half))},
                                torch::TensorOptions {}.device(atcg::CPU).dtype(torch::kFloat16))
                   .cuda();


    _mlp = MLP<1, 8, 64, 8>(context, _weights, _bias);

    NeuralBSDFData data;
    data._device_mlp = _mlp.getDeviceMLP();
    _bsdf_data_buffer.upload(&data);

    _flags = BSDFComponentType::DiffuseReflection;
}

NeuralBSDF::~NeuralBSDF() {}

void NeuralBSDF::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    const std::string ptx_bsdf_filename = "./bin/NeuralBSDF_ptx.ptx";
    auto sample_prog_group = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__sample_neuralbsdf"});
    auto eval_prog_group   = pipeline->addCallableShader({ptx_bsdf_filename, "__direct_callable__eval_neuralbsdf"});
    uint32_t sample_idx    = sbt->addCallableEntry(sample_prog_group, _bsdf_data_buffer.get());
    uint32_t eval_idx      = sbt->addCallableEntry(eval_prog_group, _bsdf_data_buffer.get());

    BSDFVPtrTable table;
    table.sampleCallIndex = sample_idx;
    table.evalCallIndex   = eval_idx;
    table.flags           = _flags;

    _vptr_table.upload(&table);

    markInitialized();
}

}    // namespace atcg
