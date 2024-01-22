#pragma once

#include <ATCG.h>

extern atcg::Application* atcg::createApplication();

namespace atcg
{

void print_statistics()
{
    atcg::host_allocator alloc_host;
    std::size_t host_bytes_allocated   = alloc_host.bytes_allocated;
    std::size_t host_bytes_deallocated = alloc_host.bytes_deallocated;

    atcg::device_allocator alloc_dev;
    std::size_t dev_bytes_allocated   = alloc_dev.bytes_allocated;
    std::size_t dev_bytes_deallocated = alloc_dev.bytes_deallocated;

    ATCG_INFO("Memory statistics: Bytes freed/Bytes allocated (Bytes leaked)");
    ATCG_INFO("Host: {0}/{1} ({2})",
              host_bytes_deallocated,
              host_bytes_allocated,
              host_bytes_allocated - host_bytes_deallocated);
    ATCG_INFO("Device: {0}/{1} ({2})",
              dev_bytes_deallocated,
              dev_bytes_allocated,
              dev_bytes_allocated - dev_bytes_deallocated);
}

int atcg_main(Application* app)
{
    app->run();

    {
        atcg::ShaderManager::destroy();
        atcg::Renderer::destroy();
    }

    return 0;
}
}    // namespace atcg

int main(int argc, char** argv)
{
    atcg::Application* app = atcg::createApplication();
    int ret                = atcg::atcg_main(app);
    delete app;
    atcg::print_statistics();
    return ret;
}