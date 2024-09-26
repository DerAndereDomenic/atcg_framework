#include <gtest/gtest.h>

#include <Core/Log.h>
#include <Core/Memory.h>

// Define a custom test environment class
class ATCGTestEnvironment : public ::testing::Environment
{
public:
    // This will be run once before any tests
    void SetUp() override
    {
        std::cout << "ATCGTestEnvironment::SetUp() called\n";

        _logger = atcg::make_ref<atcg::Logger>();
        atcg::SystemRegistry::init();
        atcg::SystemRegistry::instance()->registerSystem(_logger.get());
    }

    // This will be run once after all tests have finished
    void TearDown() override { std::cout << "ATCGTestEnvironment::TearDown() called\n"; }

private:
    atcg::ref_ptr<atcg::Logger> _logger;
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Register the global environment that will run SetUp and TearDown
    ::testing::AddGlobalTestEnvironment(new ATCGTestEnvironment);

    return RUN_ALL_TESTS();
}