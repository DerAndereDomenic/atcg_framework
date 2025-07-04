#include <gtest/gtest.h>

#include <Core/Log.h>
#include <Core/Memory.h>
#include <Core/Window.h>
#include <Renderer/ShaderManager.h>
#include <Renderer/Renderer.h>
#include <Renderer/ContextManager.h>
#include <Scene/RevisionStack.h>

// Define a custom test environment class
class ATCGTestEnvironment : public ::testing::Environment
{
public:
    // This will be run once before any tests
    void SetUp() override
    {
        std::cout << "ATCGTestEnvironment::SetUp() called\n";

        _logger = spdlog::stdout_color_mt("ATCG");
        _logger->set_pattern("%^[%H:%M:%S] %n(%l): %$%v");
        _logger->set_level(spdlog::level::trace);
        atcg::SystemRegistry::init();
        atcg::SystemRegistry::instance()->registerSystem(_logger.get());

        _context_manager = atcg::make_ref<atcg::ContextManagerSystem>();
        atcg::SystemRegistry::instance()->registerSystem(_context_manager.get());

        _shader_manager = atcg::make_ref<atcg::ShaderManagerSystem>();
        atcg::SystemRegistry::instance()->registerSystem(_shader_manager.get());

        atcg::WindowProps props;
        props.hidden = true;
        _window      = atcg::make_scope<atcg::Window>(props);

        _renderer = atcg::make_ref<atcg::RendererSystem>();
        _renderer->init(_window->getWidth(), _window->getHeight(), _window->getContext(), _shader_manager);
        atcg::SystemRegistry::instance()->registerSystem(_renderer.get());

        _revision_system = atcg::make_ref<atcg::RevisionSystem>();
        atcg::SystemRegistry::instance()->registerSystem(_revision_system.get());
    }

    // This will be run once after all tests have finished
    void TearDown() override
    {
        atcg::SystemRegistry::shutdown();
        std::cout << "ATCGTestEnvironment::TearDown() called\n";
    }

private:
    atcg::ref_ptr<spdlog::logger> _logger;
    atcg::ref_ptr<atcg::ContextManagerSystem> _context_manager;
    atcg::ref_ptr<atcg::ShaderManagerSystem> _shader_manager;
    atcg::ref_ptr<atcg::RendererSystem> _renderer;
    atcg::ref_ptr<atcg::Window> _window;
    atcg::ref_ptr<atcg::RevisionSystem> _revision_system;
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Register the global environment that will run SetUp and TearDown
    ::testing::AddGlobalTestEnvironment(new ATCGTestEnvironment);

    return RUN_ALL_TESTS();
}