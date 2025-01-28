#include <Renderer/ContextManager.h>

#include <Core/Assert.h>

#include <unordered_map>
#include <mutex>

namespace atcg
{
class ContextManagerSystem::Impl
{
public:
    Impl();

    ~Impl();

    std::unordered_map<ContextHandle, atcg::ref_ptr<atcg::Context>> context_map;

    std::mutex map_mutex;
};

ContextManagerSystem::Impl::Impl() {}

ContextManagerSystem::Impl::~Impl() {}

ContextManagerSystem::ContextManagerSystem()
{
    impl = std::make_unique<Impl>();
}

ContextManagerSystem::~ContextManagerSystem() {}

atcg::ref_ptr<atcg::Context> ContextManagerSystem::createContext(const int device_id)
{
    atcg::ref_ptr<atcg::Context> context = atcg::make_ref<atcg::Context>();
    context->create(device_id);

    std::lock_guard guard(impl->map_mutex);
    impl->context_map.insert(std::make_pair(context->getContextHandle(), context));

    return context;
}

atcg::ref_ptr<atcg::Context> ContextManagerSystem::createContext(const atcg::ref_ptr<atcg::Context>& shared)
{
    atcg::ref_ptr<atcg::Context> context = atcg::make_ref<atcg::Context>();
    context->create(shared);

    std::lock_guard guard(impl->map_mutex);
    impl->context_map.insert(std::make_pair(context->getContextHandle(), context));

    return context;
}

void ContextManagerSystem::destroyContext(atcg::ref_ptr<atcg::Context>& context)
{
    std::lock_guard guard(impl->map_mutex);
    impl->context_map.erase(context->getContextHandle());
    context->destroy();
    context = nullptr;
}

atcg::ref_ptr<atcg::Context> ContextManagerSystem::getCurrentContext() const
{
    ContextHandle current_context = Context::getCurrentContextHandle();
    std::lock_guard guard(impl->map_mutex);
    auto it = impl->context_map.find(current_context);

    ATCG_ASSERT(it != impl->context_map.end(), "Current context is not part of the Context Manager");

    return it->second;
}
}    // namespace atcg