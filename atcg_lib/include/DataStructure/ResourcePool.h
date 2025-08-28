#pragma once

#include <Renderer/Texture.h>
#include <Renderer/Framebuffer.h>

#include <unordered_map>

namespace atcg
{

struct ResourceDescription
{
    std::string name;

    FramebufferSpecification spec;
};

struct ResourceDescriptionEqual
{
    bool operator()(const ResourceDescription& lhs, const ResourceDescription& rhs) const
    {
        return lhs.name == rhs.name && lhs.spec.width == rhs.spec.width && lhs.spec.height == rhs.spec.height &&
               lhs.spec.depth == rhs.spec.depth;
    }
};

struct ResourceDescriptionHash
{
    std::size_t operator()(const ResourceDescription& res) const
    {
        std::size_t h1 = std::hash<std::string> {}(res.name);
        std::size_t h2 = std::hash<int> {}(res.spec.width);
        std::size_t h3 = std::hash<int> {}(res.spec.height);
        std::size_t h4 = std::hash<int> {}(res.spec.depth);

        // Combine the hashes (standard technique)
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

class ResourcePool
{
public:
    ResourcePool() = default;

    atcg::ref_ptr<Framebuffer> acquireFramebuffer(ResourceDescription desc);

    void garbageCollect();

private:
    struct ResourceEntry
    {
        atcg::ref_ptr<Framebuffer> fbo;
        uint32_t live_time = 0;
    };
    std::unordered_map<ResourceDescription, ResourceEntry, ResourceDescriptionHash, ResourceDescriptionEqual>
        _resources;
};
}    // namespace atcg