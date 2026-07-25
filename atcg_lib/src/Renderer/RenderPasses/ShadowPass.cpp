#include <Renderer/RenderPasses/ShadowPass.h>

#include <Renderer/Renderer.h>
#include <Scene/Components.h>
#include <Scene/ComponentRenderer.h>

namespace atcg
{
ShadowPass::ShadowPass(Dictionary& properties) : RenderPass(properties, "ShadowPass")
{
    _resolution = properties.getValueOr<uint32_t>("resolution", 1024u);
}

RenderPassReflection ShadowPass::reflect(const CompileData& ctx)
{
    RenderPassReflection reflection;

    ResourceDescription desc;
    desc.type           = ResourceType::Texture;
    desc.texture.type   = TextureType::TEXTURE_CUBE_ARRAY;
    desc.texture.depth  = TextureSizeHint::DYNAMIC;
    desc.texture.format = TextureFormat::DEPTH;

    uint32_t depth_handle = reflection.addOutput("point_light_depth_maps", desc);

    reflection.setOutputFramebufferData(_resolution, _resolution, {depth_handle});

    return reflection;
}

void ShadowPass::execute(const RenderContext& ctx, const ResourceTable& resources)
{
    auto renderer = ctx.renderer;
    auto scene    = ctx.scene;

    float n              = 0.1f;
    float f              = 100.0f;
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, n, f);

    const atcg::ref_ptr<Shader>& depth_pass_shader = renderer->getShaderManager()->getShader("depth_pass");
    depth_pass_shader->setFloat("far_plane", f);

    auto light_view = scene->getAllEntitiesWith<PointLightComponent, TransformComponent>();

    uint32_t num_lights = 0;
    for(auto e: light_view)
    {
        ++num_lights;
    }

    if(num_lights == 0)
    {
        return;
    }

    atcg::ref_ptr<TextureCubeArray> point_light_depth_maps = resources.getTexture<TextureCubeArray>("point_light_depth_"
                                                                                                    "maps");

    auto point_light_framebuffer = resources.getTargetFBO();
    if(point_light_depth_maps->getSpecification().depth != num_lights)
    {
        TextureSpecification spec                   = point_light_depth_maps->getSpecification();
        spec.depth                                  = num_lights;
        atcg::ref_ptr<TextureCubeArray> new_texture = TextureCubeArray::create(spec);
        point_light_depth_maps->swap(new_texture);    // Somewhat hacky because this dodges the resource system
        point_light_framebuffer->attachDepth(point_light_depth_maps);
    }

    GraphicsCommand::beginRenderPass(point_light_framebuffer);
    GraphicsCommand::clear();

    uint32_t light_idx = 0;
    for(auto e: light_view)
    {
        atcg::Entity entity(e, scene.get());

        auto& point_light = entity.getComponent<PointLightComponent>();
        auto& transform   = entity.getComponent<TransformComponent>();

        if(!point_light.cast_shadow)
        {
            ++light_idx;
            continue;
        }

        glm::vec3 lightPos = transform.getPosition();
        depth_pass_shader->setVec3("lightPos", lightPos);
        depth_pass_shader->setMat4(
            "shadowMatrices[0]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[1]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[2]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[3]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[4]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setMat4(
            "shadowMatrices[5]",
            projection * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
        depth_pass_shader->setInt("light_idx", light_idx);
        auto camera = ctx.camera;

        const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

        // Draw scene
        Dictionary auxiliary;
        auxiliary.setValue("override_shader", depth_pass_shader);
        for(auto e: view)
        {
            atcg::Entity entity(e, scene.get());

            renderComponent<MeshRenderComponent>(renderer, entity, camera, auxiliary);
        }

        ++light_idx;
    }

    GraphicsCommand::endRenderPass();
}

void ShadowPass::registerRenderPass(RenderPassRegistry::Registry* registry)
{
    ATCG_REGISTER_RENDER_PASS(registry, "ShadowPass", ShadowPass);
}


}    // namespace atcg