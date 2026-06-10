#pragma once

#include <Core/API.h>
#include <DataStructure/GPUResource.h>

#include <unordered_map>

namespace atcg
{
/**
 * @brief A class to model the resource table passed to the render passes.
 */
class ATCG_API ResourceTable
{
public:
    /**
     * @brief Get a resource from the table by name.
     *
     * @param name The name of the resource to retrieve.
     * @return The requested resource, or a default-constructed one if not found.
     */
    PhysicalResource operator[](const std::string& name) const
    {
        auto it = _resources.find(name);
        if(it != _resources.end())
            return it->second;
        else
            return PhysicalResource();
    }

    /**
     * @brief Get a texture resource from the table by name. Returns nullptr if the resource is not found or if it is
     * not a texture.
     *
     * @tparam TextureType The expected type of the texture.
     * @param name The name of the texture to retrieve.
     * @return The requested texture, or nullptr if not found or not a texture.
     */
    template<typename TextureType = Texture>
    atcg::ref_ptr<TextureType> getTexture(const std::string& name) const
    {
        auto resource = this->operator[](name);

        if(std::holds_alternative<atcg::ref_ptr<Texture>>(resource))
        {
            return std::dynamic_pointer_cast<TextureType>(std::get<atcg::ref_ptr<Texture>>(resource));
        }

        return nullptr;
    }

    /**
     * @brief Get a tensor resource from the table by name. Returns an empty tensor if the resource is not found or if
     * it is not a tensor.
     *
     * @param name The name of the tensor to retrieve.
     * @return The requested tensor, or an empty tensor if not found or not a tensor
     */
    torch::Tensor getTensor(const std::string& name) const
    {
        auto resource = this->operator[](name);

        if(std::holds_alternative<torch::Tensor>(resource))
        {
            return std::get<torch::Tensor>(resource);
        }

        return torch::Tensor();
    }

    /**
     * @brief Get a vertex buffer resource from the table by name. Returns nullptr if the resource is not found or if it
     * is not a vertex buffer.
     *
     * @param name The name of the vertex buffer to retrieve.
     * @return The requested vertex buffer, or nullptr if not found or not a vertex buffer
     */
    atcg::ref_ptr<VertexBuffer> getBuffer(const std::string& name) const
    {
        auto resource = this->operator[](name);

        if(std::holds_alternative<atcg::ref_ptr<VertexBuffer>>(resource))
        {
            return std::get<atcg::ref_ptr<VertexBuffer>>(resource);
        }

        return nullptr;
    }

    /**
     * @brief Set a resource in the table by name.
     *
     * @param name The name of the resource to set.
     * @param resource The resource to set.
     */
    void set(const std::string& name, const PhysicalResource& resource) { _resources[name] = resource; }

    /**
     * @brief Get the target framebuffer for this resource table
     *
     * @return The target framebuffer
     */
    atcg::ref_ptr<Framebuffer> getTargetFBO() const { return _target_fbo; }

    /**
     * @brief Set the target framebuffer
     *
     * @param fbo The target framebuffer
     */
    void setTargetFBO(const atcg::ref_ptr<Framebuffer>& fbo) { _target_fbo = fbo; }

private:
    std::unordered_map<std::string, PhysicalResource> _resources;
    atcg::ref_ptr<Framebuffer> _target_fbo;
};
}    // namespace atcg