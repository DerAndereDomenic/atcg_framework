#include <Utils/Utils.h>

#include <Asset/Project.h>
#include <Renderer/Renderer.h>

#include <fstream>

namespace atcg
{

namespace Utils
{
void normalize(const atcg::ref_ptr<Graph>& graph)
{
    auto vertices = graph->getPositions(atcg::GPU);

    auto max_scale  = torch::amax(torch::abs(vertices));
    auto mean_point = torch::mean(vertices, 0);

    vertices -= mean_point;
    vertices /= max_scale;
}

void normalize(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform)
{
    auto vertices = graph->getPositions(atcg::GPU);

    auto max_scale  = torch::amax(torch::abs(vertices));
    auto mean_point = torch::mean(vertices, 0);

    vertices -= mean_point;
    vertices /= max_scale;

    glm::mat4 model = transform.getModel();

    glm::vec3 mean_vector = glm::make_vec3((float*)mean_point.cpu().contiguous().data_ptr());

    model = model * glm::translate(mean_vector) * glm::scale(glm::vec3(max_scale.item<float>()));

    transform.setModel(model);
}

void applyTransform(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform)
{
    auto vertices = graph->getPositions(atcg::GPU);
    auto normals  = graph->getNormals(atcg::GPU);
    auto tangents = graph->getTangents(atcg::GPU);

    applyTransform(vertices, normals, tangents, transform);

    transform.setModel(glm::mat4(1));
}

void applyTransform(torch::Tensor& vertices,
                    torch::Tensor& normals,
                    torch::Tensor& tangents,
                    atcg::TransformComponent& transform)
{
    glm::mat4 model_matrix  = transform.getModel();
    glm::mat4 normal_matrix = glm::inverse(glm::transpose(model_matrix));

    torch::Tensor model_tensor  = atcg::createHostTensorFromPointer(glm::value_ptr(model_matrix), {4, 4});
    torch::Tensor normal_tensor = atcg::createHostTensorFromPointer(glm::value_ptr(normal_matrix), {4, 4});
    auto options                = atcg::TensorOptions::HostOptions<float>();
    if(vertices.is_cuda())
    {
        model_tensor  = model_tensor.cuda();
        normal_tensor = normal_tensor.cuda();
        options       = atcg::TensorOptions::DeviceOptions<float>();
    }

    torch::Tensor ones  = torch::ones({vertices.size(0), 1}, options);
    torch::Tensor zeros = torch::zeros({vertices.size(0), 1}, options);

    auto vertices_hom = torch::hstack({vertices, ones});
    auto normals_hom  = torch::hstack({normals, zeros});
    auto tangents_hom = torch::hstack({tangents, zeros});

    vertices_hom = torch::matmul(vertices_hom, model_tensor);
    normals_hom  = torch::matmul(normals_hom, normal_tensor);
    tangents_hom = torch::matmul(tangents_hom, normal_tensor);
    normals_hom  = normals_hom / (torch::norm(normals_hom, 2, -1, true) + 1e-5f);
    tangents_hom = tangents_hom / (torch::norm(tangents_hom, 2, -1, true) + 1e-5f);

    vertices.index_put_({torch::indexing::Slice(), torch::indexing::Slice()},
                        vertices_hom.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));
    normals.index_put_({torch::indexing::Slice(), torch::indexing::Slice()},
                       normals_hom.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));
    tangents.index_put_({torch::indexing::Slice(), torch::indexing::Slice()},
                        tangents_hom.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));
}

template<>
int16_t ntoh<int16_t>(int16_t network)
{
    if(isLittleEndian())
    {
        unsigned char* data = (unsigned char*)&network;
        return (data[1] << 0) | ((unsigned)data[0] << 8);
    }
    return network;
}

template<>
uint16_t ntoh<uint16_t>(uint16_t network)
{
    if(isLittleEndian())
    {
        unsigned char* data = (unsigned char*)&network;
        return (data[1] << 0) | ((unsigned)data[0] << 8);
    }
    return network;
}

template<>
int32_t ntoh<int32_t>(int32_t network)
{
    if(isLittleEndian())
    {
        unsigned char* data = (unsigned char*)&network;
        return (data[3] << 0) | (data[2] << 8) | (data[1] << 16) | ((unsigned)data[0] << 24);
    }
    return network;
}

template<>
uint32_t ntoh<uint32_t>(uint32_t network)
{
    if(isLittleEndian())
    {
        unsigned char* data = (unsigned char*)&network;
        return (data[3] << 0) | (data[2] << 8) | (data[1] << 16) | ((unsigned)data[0] << 24);
    }
    return network;
}

template<>
int64_t ntoh<int64_t>(int64_t network)
{
    if(isLittleEndian())
    {
        unsigned char* data = (unsigned char*)&network;
        return ((int64_t)data[7] << 0) | ((int64_t)data[6] << 8) | ((int64_t)data[5] << 16) | ((int64_t)data[4] << 24) |
               ((int64_t)data[3] << 32) | ((int64_t)data[2] << 40) | ((int64_t)data[1] << 48) |
               ((int64_t)(unsigned)data[0] << 56);
    }
    return network;
}

template<>
uint64_t ntoh<uint64_t>(uint64_t network)
{
    if(isLittleEndian())
    {
        unsigned char* data = (unsigned char*)&network;
        return ((uint64_t)data[7] << 0) | ((uint64_t)data[6] << 8) | ((uint64_t)data[5] << 16) |
               ((uint64_t)data[4] << 24) | ((uint64_t)data[3] << 32) | ((uint64_t)data[2] << 40) |
               ((uint64_t)data[1] << 48) | ((uint64_t)(unsigned)data[0] << 56);
    }
    return network;
}

void dumpBinary(const std::string& path, const torch::Tensor& data)
{
    auto data_ = data.to(atcg::CPU);
    std::ofstream out(path, std::ios::out | std::ios::binary);
    out.write((const char*)data_.data_ptr(), data_.numel() * data_.element_size());
}

void screenshot(const atcg::ref_ptr<Scene>& scene,
                const atcg::ref_ptr<Camera>& camera,
                const uint32_t width,
                const std::string& path)
{
    auto data = screenshot(scene, camera, width);

    Image img(data);

    img.store(path);
}

void screenshot(const atcg::ref_ptr<Scene>& scene,
                const atcg::ref_ptr<Camera>& camera,
                const uint32_t width,
                const uint32_t height,
                const std::string& path)
{
    atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
    screenshot_buffer->attachColor();
    screenshot_buffer->attachDepth();
    screenshot_buffer->complete();

    atcg::Dictionary context;
    context.setValue("camera", camera);
    context.setValue("target", screenshot_buffer);
    scene->draw(context);

    auto data = screenshot_buffer->getColorAttachement(0)->getData(atcg::CPU);

    Image img(data);

    img.store(path);
}

torch::Tensor screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width)
{
    float height                                 = (float)width / camera->getIntrinsics().aspectRatio();
    atcg::ref_ptr<Framebuffer> screenshot_buffer = atcg::make_ref<Framebuffer>((int)width, (int)height);
    screenshot_buffer->attachColor();
    screenshot_buffer->attachDepth();
    screenshot_buffer->complete();

    atcg::Dictionary context;
    context.setValue("camera", camera);
    context.setValue("target", screenshot_buffer);
    scene->draw(context);

    auto data = screenshot_buffer->getColorAttachement(0)->getData(atcg::CPU);

    return data;
}

Entity pickEntity(const glm::vec2& mouse_pos)
{
    auto fbo        = atcg::Renderer::getFramebuffer();
    auto pixel_data = fbo->getColorAttachement(1)->getData(
        atcg::CPU);    // TODO: Overkill to copy the entire buffer just for one pixel

    int pixelData = pixel_data.index({(int)mouse_pos.y, (int)mouse_pos.x, 0}).item<int>();

    return pixelData == -1 ? atcg::Entity()
                           : atcg::Entity((entt::entity)pixelData, atcg::Project::getActive()->getActiveScene().get());
}

uint32_t setLights(atcg::RendererSystem* renderer,
                   Scene* scene,
                   const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                   const atcg::ref_ptr<Shader>& shader)
{
    auto light_view = scene->getAllEntitiesWith<atcg::PointLightComponent, atcg::TransformComponent>();

    uint32_t num_lights = 0;
    for(auto e: light_view)
    {
        std::stringstream light_index;
        light_index << "[" << num_lights << "]";
        std::string light_index_str = light_index.str();

        atcg::Entity light_entity(e, scene);

        auto& point_light     = light_entity.getComponent<atcg::PointLightComponent>();
        auto& light_transform = light_entity.getComponent<atcg::TransformComponent>();

        shader->setVec3("light_colors" + light_index_str, point_light.color);
        shader->setFloat("light_intensities" + light_index_str, point_light.intensity);
        shader->setVec3("light_positions" + light_index_str, light_transform.getPosition());

        ++num_lights;
    }

    shader->setInt("num_lights", num_lights);
    if(point_light_depth_maps)
    {
        uint32_t shadow_map_id = renderer->popTextureID();
        shader->setInt("shadow_maps", shadow_map_id);
        shader->setInt("shadow_pass", 1);
        GraphicsCommand::bindTexture(shadow_map_id, point_light_depth_maps);

        return shadow_map_id;
    }
    else
    {
        shader->setInt("shadow_pass", 0);
        //     ATCG_ASSERT(num_lights == 0, "Shadow map is not initialized but lights are present");
    }

    return -1;
}

std::pair<uint32_t, uint32_t>
setSkyLight(atcg::RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader, const atcg::ref_ptr<Skybox>& skybox)
{
    uint32_t irradiance_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(irradiance_id, skybox->getIrradianceMap());
    shader->setInt("irradiance_map", irradiance_id);

    uint32_t prefiltered_id = renderer->popTextureID();
    GraphicsCommand::bindTexture(prefiltered_id, skybox->getPrefilteredMap());
    shader->setInt("prefilter_map", prefiltered_id);

    return std::make_pair(irradiance_id, prefiltered_id);
}

AssetHandle displayMaterialSelection(const std::string& key, AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "Default Material";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Material##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("Default Material", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Material) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
#else
    return 0;
#endif
}

AssetHandle displayGraphSelection(const std::string& key, AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Graph";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Graph##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Graph", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Graph) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
#else
    return 0;
#endif
}

AssetHandle displayScriptSelection(const std::string& key, AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Script";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Script##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Script", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Script) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
#else
    return 0;
#endif
}

AssetHandle displayShaderSelection(const std::string& key, AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "Default Shader";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Shader##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("Default Shader", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Shader) continue;

            auto shader = AssetManager::getAsset<Shader>(it->first);
            if(shader && shader->isComputeShader()) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
#else
    return 0;
#endif
}

AssetHandle displayTexture2DSelection(const std::string& key, AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Image";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Image##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Image", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Texture2D) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
#else
    return 0;
#endif
}

AssetHandle displayTexture3DSelection(const std::string& key, AssetHandle handle)
{
#ifndef ATCG_HEADLESS
    const auto& data = AssetManager::getMetaData(handle);

    std::string tag = AssetManager::isAssetHandleValid(handle) ? data.name : "No Image";

    const auto& registry = AssetManager::getAssetRegistry();

    AssetHandle current_item = handle;

    if(ImGui::BeginCombo(("Select Image##" + key).c_str(), tag.c_str()))
    {
        // No Selection
        {
            bool is_selected = !AssetManager::isAssetHandleValid(current_item);

            if(ImGui::Selectable("No Image", is_selected))
            {
                current_item = 0;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        for(auto it = registry.begin(); it != registry.end(); ++it)
        {
            if(it->second.type != AssetType::Texture3D) continue;

            bool is_selected = it->first == current_item;

            if(ImGui::Selectable((it->second.name + "##" + std::to_string(it->first)).c_str(), is_selected))
            {
                current_item = it->first;
            }

            if(is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    return current_item;
#else
    return 0;
#endif
}

void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size)
{
    std::ofstream summary_file(file_name, std::ios::out | std::ios::binary);
    summary_file.write(data, byte_size);
    summary_file.close();
}

std::vector<uint8_t> deserializeBuffer(const std::string& file_name)
{
    std::ifstream summary_file(file_name, std::ios::in | std::ios::binary);
    std::vector<uint8_t> buffer_char(std::istreambuf_iterator<char>(summary_file), {});
    summary_file.close();

    return buffer_char;
}

nlohmann::json serializeLayout(const atcg::BufferLayout& layout)
{
    nlohmann::json::array_t json_layout;
    for(auto element: layout)
    {
        nlohmann::json::array_t json_element;
        json_element.push_back((int)element.type);
        json_element.push_back(element.name);

        json_layout.push_back(json_element);
    }

    return json_layout;
}

atcg::BufferLayout deserializeLayout(nlohmann::json& layout_node)
{
    std::vector<atcg::BufferElement> elements;
    for(nlohmann::json::array_t element: layout_node)
    {
        atcg::BufferElement buffer_element((atcg::ShaderDataType)element[0], element[1]);
        elements.push_back(buffer_element);
    }

    return atcg::BufferLayout(elements);
}

}    // namespace Utils
}    // namespace atcg