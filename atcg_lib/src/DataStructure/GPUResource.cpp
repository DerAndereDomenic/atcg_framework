#include <DataStructure/GPUResource.h>

namespace atcg
{
PhysicalResource createResource(const CompileData& ctx, const ResourceDescription& desc)
{
    switch(desc.type)
    {
        case ResourceType::Buffer:
        {
            atcg::ref_ptr<VertexBuffer> buffer = atcg::make_ref<VertexBuffer>(desc.buffer.size);
            buffer->setLayout(desc.buffer.layout);
            return PhysicalResource(buffer);
        }
        break;
        case ResourceType::Texture:
        {
            TextureSpecification spec;
            spec.width       = desc.texture.width;
            spec.height      = desc.texture.height;
            spec.num_samples = ctx.num_samples;
            spec.depth       = desc.texture.depth.convert(1);
            spec.sampler     = desc.texture.sampler;
            spec.format      = desc.texture.format;
            return Texture::create(desc.texture.type, spec);
        }
        break;
        case ResourceType::Tensor:
        {
            return torch::tensor(desc.tensor.size, desc.tensor.options);
        }
        break;
    }

    // Just return something
    return torch::Tensor();
}
}    // namespace atcg