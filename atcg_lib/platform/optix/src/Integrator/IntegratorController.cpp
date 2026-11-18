#include <Integrator/IntegratorController.h>

#include <Asset/Project.h>
#include <Core/RaytracingContextManager.h>
#include <Integrator/IntegratorRegistry.h>
#include <Renderer/Renderer.h>
#include <Scene/RevisionStack.h>

namespace atcg
{

namespace detail
{
/**
 * @brief Get the integrator settings of the active project. If the project has no settings, settings for the first
 * registered integrator are created.
 *
 * @return The settings or nullptr if there is no active project or no registered integrator
 */
atcg::ref_ptr<IntegratorSettingsBase> ensureIntegratorSettings()
{
    const auto& project = Project::getActive();
    if(!project)
    {
        ATCG_ERROR("Path tracing requires an active project");
        return nullptr;
    }

    if(!project->getIntegratorSettings())
    {
        const auto& types = IntegratorRegistry::getRegistry()->getRegisteredTypes();
        if(types.empty())
        {
            ATCG_ERROR("Path tracing requires a registered integrator");
            return nullptr;
        }
        project->setIntegratorSettings(IntegratorRegistry::createIntegratorSettings(types.front()));
    }

    return project->getIntegratorSettings();
}
}    // namespace detail

class IntegratorController::Impl
{
public:
    bool rebuild(const atcg::ref_ptr<Scene>& scene,
                 const atcg::ref_ptr<IntegratorSettingsBase>& settings,
                 uint32_t width,
                 uint32_t height);

    atcg::ref_ptr<RaytracingContext> context;
    atcg::ref_ptr<Integrator> integrator;

    atcg::ref_ptr<Texture2D> output_texture;
    atcg::ref_ptr<Texture2D> output_entity_texture;

    bool running = false;
    bool dirty   = true;

    // State of the last frame to detect changes. Weak pointers are used so that a new object at the address of a
    // destroyed one is not mistaken for the old one.
    std::weak_ptr<Scene> last_scene;
    std::weak_ptr<IntegratorSettingsBase> last_settings;
    uint32_t last_width       = 0;
    uint32_t last_height      = 0;
    uint32_t last_revision    = 0;
    glm::mat4 last_view       = glm::mat4(0);
    glm::mat4 last_projection = glm::mat4(0);
};

bool IntegratorController::Impl::rebuild(const atcg::ref_ptr<Scene>& scene,
                                         const atcg::ref_ptr<IntegratorSettingsBase>& settings,
                                         uint32_t width,
                                         uint32_t height)
{
    // Release the old integrator first so that its device memory is freed before the new one is allocated
    integrator = nullptr;

    atcg::Dictionary dict = settings->toDictionary();
    // The template argument is deduced (forwarding reference), the stored types are ref_ptr<Scene> and uint32_t
    dict.setValue("scene", scene);
    dict.setValue("width", width);
    dict.setValue("height", height);

    integrator = IntegratorRegistry::createIntegrator(settings->getType(), context, dict);
    if(!integrator)
    {
        return false;
    }

    TextureSpecification spec;
    spec.width     = width;
    spec.height    = height;
    spec.format    = TextureFormat::RGBA;
    output_texture = Texture2D::create(spec);

    TextureSpecification spec_int;
    spec_int.width        = width;
    spec_int.height       = height;
    spec_int.format       = TextureFormat::RINT;
    output_entity_texture = Texture2D::create(spec_int);

    return true;
}

IntegratorController::IntegratorController(const atcg::ref_ptr<RaytracingContext>& context)
{
    impl          = std::make_unique<Impl>();
    impl->context = context ? context : RaytracingContextManager::createContext();

    // Create default settings so that they can be edited before path tracing is started
    if(Project::getActive())
    {
        detail::ensureIntegratorSettings();
    }
}

IntegratorController::~IntegratorController() {}

void IntegratorController::start()
{
    if(!detail::ensureIntegratorSettings())
    {
        return;
    }

    impl->running = true;
    impl->dirty   = true;
}

void IntegratorController::stop()
{
    impl->running = false;
}

bool IntegratorController::isRunning() const
{
    return impl->running;
}

void IntegratorController::setIntegratorType(const std::string& type)
{
    const auto& project = Project::getActive();
    if(!project)
    {
        ATCG_ERROR("Cannot set the integrator type without an active project");
        return;
    }

    auto settings = IntegratorRegistry::createIntegratorSettings(type);
    if(!settings)
    {
        return;
    }

    project->setIntegratorSettings(settings);
    impl->dirty = true;
}

std::string IntegratorController::getIntegratorType() const
{
    auto settings = getSettings();
    return settings ? settings->getType() : std::string();
}

atcg::ref_ptr<IntegratorSettingsBase> IntegratorController::getSettings() const
{
    const auto& project = Project::getActive();
    return project ? project->getIntegratorSettings() : nullptr;
}

void IntegratorController::requestRebuild()
{
    impl->dirty = true;
}

void IntegratorController::reset()
{
    if(impl->integrator)
    {
        impl->integrator->reset();
    }
}

void IntegratorController::release()
{
    impl->integrator = nullptr;
    impl->dirty      = true;
}

atcg::ref_ptr<Integrator> IntegratorController::getIntegrator() const
{
    return impl->integrator;
}

bool IntegratorController::render(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Framebuffer>& target)
{
    if(!impl->running || !scene || !target)
    {
        return false;
    }

    // A new or loaded project might not have integrator settings
    auto settings = detail::ensureIntegratorSettings();
    if(!settings)
    {
        impl->running = false;
        return false;
    }

    uint32_t width    = target->width();
    uint32_t height   = target->height();
    uint32_t revision = RevisionStack::numUndos();

    // Rebuild the integrator if anything changed that affects the optix scene
    bool needs_rebuild = impl->dirty || !impl->integrator || impl->last_scene.lock() != scene ||
                         impl->last_settings.lock() != settings || impl->last_width != width ||
                         impl->last_height != height || impl->last_revision != revision;

    if(needs_rebuild)
    {
        impl->dirty         = false;
        impl->last_scene    = scene;
        impl->last_settings = settings;
        impl->last_width    = width;
        impl->last_height   = height;
        impl->last_revision = revision;

        if(!impl->rebuild(scene, settings, width, height))
        {
            ATCG_ERROR("Could not create integrator {}. Path tracing is stopped.", settings->getType());
            impl->running = false;
            return false;
        }
    }

    // Reset the accumulation if the camera moved
    auto camera = scene->getCamera();
    if(camera)
    {
        glm::mat4 view       = camera->getView();
        glm::mat4 projection = camera->getProjection();
        if(view != impl->last_view || projection != impl->last_projection)
        {
            impl->last_view       = view;
            impl->last_projection = projection;
            if(!needs_rebuild)
            {
                impl->integrator->reset();
            }
        }
    }

    atcg::Dictionary dict;
    impl->integrator->generateRays(dict);
    impl->output_texture->setData(dict.getValue<torch::Tensor>("output"));
    impl->output_entity_texture->setData(dict.getValue<torch::Tensor>("entity_ids"));

    GraphicsCommand::beginRenderPass(target);
    GraphicsCommand::clear();
    Renderer::drawImage(impl->output_texture, impl->output_entity_texture);
    GraphicsCommand::endRenderPass();

    return true;
}

atcg::ref_ptr<Texture2D> IntegratorController::getOutputTexture() const
{
    return impl->output_texture;
}

atcg::ref_ptr<Texture2D> IntegratorController::getOutputEntityTexture() const
{
    return impl->output_entity_texture;
}

}    // namespace atcg
