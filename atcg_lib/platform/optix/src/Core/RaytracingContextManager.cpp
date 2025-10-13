#include <Core/RaytracingContextManager.h>

#include <Core/Assert.h>

#include <unordered_map>
#include <mutex>

namespace atcg
{
class RaytracingContextManagerSystem::Impl
{
public:
    Impl();

    ~Impl();

    std::unordered_map<uint64_t, atcg::ref_ptr<atcg::RaytracingContext>> context_map;

    std::mutex map_mutex;
};

RaytracingContextManagerSystem::Impl::Impl() {}

RaytracingContextManagerSystem::Impl::~Impl() {}

RaytracingContextManagerSystem::RaytracingContextManagerSystem()
{
    impl = std::make_unique<Impl>();
}

RaytracingContextManagerSystem::~RaytracingContextManagerSystem() {}

atcg::ref_ptr<atcg::RaytracingContext> RaytracingContextManagerSystem::createContext(const int device_id)
{
    atcg::ref_ptr<atcg::RaytracingContext> context =
        atcg::ref_ptr<atcg::RaytracingContext>(new atcg::RaytracingContext());
    context->initRaytracingAPI();    // The first context created this way, will enable the optix API
    context->create(device_id);

    std::lock_guard guard(impl->map_mutex);
    impl->context_map.insert(std::make_pair((uint64_t)context->getContextHandle(), context));

    return context;
}

void RaytracingContextManagerSystem::destroyContext(atcg::ref_ptr<atcg::RaytracingContext>& context)
{
    std::lock_guard guard(impl->map_mutex);
    impl->context_map.erase((uint64_t)context->getContextHandle());
    context->destroy();
    context = nullptr;
}
}    // namespace atcg