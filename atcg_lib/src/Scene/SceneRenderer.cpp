#include <Scene/SceneRenderer.h>

#include <Core/Assert.h>
#include <Scene/Components.h>
#include <Scene/Entity.h>
#include <Scene/ComponentRenderer.h>
#include <Renderer/RenderGraph.h>

namespace atcg
{

class SceneRenderer::Impl
{
public:
    Impl();

    ~Impl();

    Dictionary context;

    atcg::ref_ptr<atcg::RenderGraph> graph;
};

SceneRenderer::Impl::Impl()
{
    graph = atcg::make_ref<atcg::RenderGraph>();

    auto [skybox_handle, skybox_builder] = graph->addRenderPass("Skybox");    // Outputs into the current render target
    auto [shadow_handle, shadow_builder] = graph->addRenderPass("ShadowMaps");
    auto [output_handle, output_builder] = graph->addRenderPass("Forward");

    // SKYBOX PASS
    skybox_builder->registerOutput("framebuffer", nullptr)
        ->setRenderFunction([](Dictionary& context, const Dictionary&, Dictionary&, Dictionary&)
                            { atcg::Renderer::drawSkybox(context.getValue<atcg::ref_ptr<Camera>>("camera")); });

    // SHADOW PASS
    shadow_builder
        ->registerOutput("point_light_depth_maps", atcg::make_ref<atcg::ref_ptr<atcg::TextureCubeArray>>(nullptr))
        ->setSetupFunction([](Dictionary&, Dictionary& data, Dictionary& output_data)
                           { data.setValue("point_light_framebuffer", atcg::make_ref<atcg::Framebuffer>(1024, 1024)); })
        ->setRenderFunction(
            [](Dictionary& context, const Dictionary&, Dictionary& data, Dictionary& output_data)
            {
                auto scene = context.getValue<atcg::ref_ptr<Scene>>("scene");

                float n              = 0.1f;
                float f              = 100.0f;
                glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, n, f);

                const atcg::ref_ptr<Shader>& depth_pass_shader = atcg::ShaderManager::getShader("depth_pass");
                depth_pass_shader->setFloat("far_plane", f);

                uint32_t active_fbo = atcg::Framebuffer::currentFramebuffer();

                glm::vec4 old_viewport = atcg::Renderer::getViewport();

                auto light_view = scene->getAllEntitiesWith<PointLightComponent, TransformComponent>();

                uint32_t num_lights = 0;
                for(auto e: light_view)
                {
                    ++num_lights;
                }

                auto point_light_depth_maps =
                    output_data.getValue<atcg::ref_ptr<atcg::ref_ptr<atcg::TextureCubeArray>>>("point_light_"
                                                                                               "depth_maps");
                if(num_lights == 0)
                {
                    *point_light_depth_maps = nullptr;
                    return;
                }

                auto point_light_framebuffer = data.getValue<atcg::ref_ptr<atcg::Framebuffer>>("point_light_"
                                                                                               "framebuffer");
                if(!(*point_light_depth_maps) || (*point_light_depth_maps)->depth() != num_lights)
                {
                    atcg::TextureSpecification spec;
                    spec.depth              = num_lights;
                    spec.width              = 1024;
                    spec.height             = 1024;
                    spec.format             = atcg::TextureFormat::DEPTH;
                    *point_light_depth_maps = atcg::TextureCubeArray::create(spec);

                    point_light_framebuffer->attachDepth(*point_light_depth_maps);
                    point_light_framebuffer->complete();
                }

                point_light_framebuffer->use();
                atcg::Renderer::setViewport(0, 0, point_light_framebuffer->width(), point_light_framebuffer->height());
                atcg::Renderer::clear();

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
                    depth_pass_shader->setMat4("shadowMatrices[0]",
                                               projection * glm::lookAt(lightPos,
                                                                        lightPos + glm::vec3(1.0, 0.0, 0.0),
                                                                        glm::vec3(0.0, -1.0, 0.0)));
                    depth_pass_shader->setMat4("shadowMatrices[1]",
                                               projection * glm::lookAt(lightPos,
                                                                        lightPos + glm::vec3(-1.0, 0.0, 0.0),
                                                                        glm::vec3(0.0, -1.0, 0.0)));
                    depth_pass_shader->setMat4("shadowMatrices[2]",
                                               projection * glm::lookAt(lightPos,
                                                                        lightPos + glm::vec3(0.0, 1.0, 0.0),
                                                                        glm::vec3(0.0, 0.0, 1.0)));
                    depth_pass_shader->setMat4("shadowMatrices[3]",
                                               projection * glm::lookAt(lightPos,
                                                                        lightPos + glm::vec3(0.0, -1.0, 0.0),
                                                                        glm::vec3(0.0, 0.0, -1.0)));
                    depth_pass_shader->setMat4("shadowMatrices[4]",
                                               projection * glm::lookAt(lightPos,
                                                                        lightPos + glm::vec3(0.0, 0.0, 1.0),
                                                                        glm::vec3(0.0, -1.0, 0.0)));
                    depth_pass_shader->setMat4("shadowMatrices[5]",
                                               projection * glm::lookAt(lightPos,
                                                                        lightPos + glm::vec3(0.0, 0.0, -1.0),
                                                                        glm::vec3(0.0, -1.0, 0.0)));
                    depth_pass_shader->setInt("light_idx", light_idx);

                    const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent,
                                                                 atcg::GeometryComponent,
                                                                 atcg::MeshRenderComponent>();

                    // Draw scene
                    for(auto e: view)
                    {
                        atcg::Entity entity(e, scene.get());

                        auto& transform = entity.getComponent<atcg::TransformComponent>();
                        auto& geometry  = entity.getComponent<atcg::GeometryComponent>();

                        atcg::Renderer::draw(geometry.graph,
                                             context.getValue<atcg::ref_ptr<Camera>>("camera"),
                                             transform.getModel(),
                                             glm::vec3(1),
                                             depth_pass_shader);
                    }

                    ++light_idx;
                }

                atcg::Renderer::setViewport(old_viewport[0], old_viewport[1], old_viewport[2], old_viewport[3]);
                atcg::Framebuffer::bindByID(active_fbo);
            });

    output_builder->setRenderFunction(
        [](Dictionary& context, const Dictionary& inputs, Dictionary&, Dictionary&)
        {
            auto scene       = context.getValue<atcg::ref_ptr<Scene>>("scene");
            auto camera      = context.getValue<atcg::ref_ptr<Camera>>("camera");
            const auto& view = scene->getAllEntitiesWith<atcg::TransformComponent>();

            atcg::ref_ptr<atcg::TextureCubeArray> point_light_depth_maps = nullptr;

            if(inputs.contains("point_light_depth_maps"))
            {
                point_light_depth_maps = *inputs.getValue<atcg::ref_ptr<atcg::ref_ptr<atcg::TextureCubeArray>>>("point_"
                                                                                                                "light_"
                                                                                                                "depth_"
                                                                                                                "maps");
            }

            Dictionary auxiliary;
            auxiliary.setValue("point_light_depth_maps", point_light_depth_maps);

            ComponentRenderer component_renderer;
            for(auto e: view)
            {
                Entity entity(e, scene.get());
                if(entity.hasComponent<CustomRenderComponent>())
                {
                    CustomRenderComponent renderer = entity.getComponent<CustomRenderComponent>();
                    renderer.callback(entity, camera);
                }

                component_renderer.renderComponent<MeshRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<PointRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<PointSphereRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<EdgeRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<EdgeCylinderRenderComponent>(entity, camera, auxiliary);
                component_renderer.renderComponent<InstanceRenderComponent>(entity, camera, auxiliary);
            }
        });


    graph->addDependency(skybox_handle, "framebuffer", output_handle, "framebuffer");
    graph->addDependency(shadow_handle, "point_light_depth_maps", output_handle, "point_light_depth_maps");

    graph->compile(context);
}

SceneRenderer::Impl::~Impl() {}

SceneRenderer::SceneRenderer()
{
    impl = std::make_unique<Impl>();
}

SceneRenderer::SceneRenderer(const atcg::ref_ptr<atcg::Scene>& scene)
{
    impl = std::make_unique<Impl>();
    setScene(scene);
}

SceneRenderer::~SceneRenderer() {}

void SceneRenderer::setScene(const atcg::ref_ptr<atcg::Scene>& scene)
{
    impl->context.setValue("scene", scene);
}

void SceneRenderer::render(const atcg::ref_ptr<Camera>& camera)
{
    impl->context.setValue("camera", camera);
    impl->graph->execute(impl->context);
}
}    // namespace atcg