#include <DataStructure/ResourcePool.h>
#include <Core/Application.h>

#define MAX_LIVE_TIME 1

namespace atcg
{
atcg::ref_ptr<Framebuffer> ResourcePool::acquireFramebuffer(ResourceDescription desc)
{
    auto it = _resources.find(desc);
    if(it == _resources.end())
    {
        ATCG_TRACE("Allocated new Framebuffer resource {} with resolution {} x {} x {}",
                   desc.name,
                   desc.spec.width,
                   desc.spec.height,
                   desc.spec.depth);
        auto fbo = Framebuffer::create(desc.spec);
        _resources.insert(std::make_pair(desc, ResourceEntry {fbo, Application::get()->getApplicationCounter()}));
        return fbo;
    }

    it->second.live_time = Application::get()->getApplicationCounter();
    return it->second.fbo;
}

void ResourcePool::garbageCollect()
{
    uint64_t current_counter = Application::get()->getApplicationCounter();
    for(auto it = _resources.begin(); it != _resources.end();)
    {
        uint64_t live_time = current_counter - it->second.live_time;

        if(live_time > MAX_LIVE_TIME)
        {
            ATCG_TRACE("Deleted Framebuffer resource {} with resolution {} x {} x {}",
                       it->first.name,
                       it->first.spec.width,
                       it->first.spec.height,
                       it->first.spec.depth);
            it = _resources.erase(it);
            continue;
        }

        ++it;
    }
}
}    // namespace atcg