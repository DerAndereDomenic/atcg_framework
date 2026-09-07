#include <Renderer/RenderPasses/OutputPass.h>

namespace atcg
{

OutputPass::OutputPass(Dictionary& properties) : RenderPass(properties, "OutputPass")
{
    _output_fbo = properties.getValueOr<atcg::ref_ptr<Framebuffer>>("output_fbo", nullptr);
}

RenderPassReflection OutputPass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;

    reflection.addInput("color");
    reflection.addInput("entities");
    reflection.addInput("stencil");
    reflection.addInput("depth");

    // Output is a "sink", all data is written to the specified output framebuffer

    return reflection;
}

void OutputPass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto color   = resources.getTexture<Texture2D>("color");
    auto entity  = resources.getTexture<Texture2D>("entities");
    auto stencil = resources.getTexture<Texture2D>("stencil");
    auto depth   = resources.getTexture<Texture2D>("depth");

    // TODO
    _output_fbo->detachAllAttachements();
    _output_fbo->attachTexture(color);
    _output_fbo->attachTexture(entity);
    _output_fbo->attachTexture(stencil);
    _output_fbo->attachDepth(depth);
    _output_fbo->complete();
}

void OutputPass::registerRenderPass(RenderPassRegistry::Registry* registry)
{
    ATCG_REGISTER_RENDER_PASS(registry, "OutputPass", OutputPass);
}


}    // namespace atcg