#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <DataStructure/Graph.h>
#include <Scene/Components.h>

#include <numeric>

namespace atcg
{
namespace Utils
{
/**
 * @brief Normalizes a graph to the unit cube
 *
 * @param graph The graph to normalize
 */
ATCG_API void normalize(const atcg::ref_ptr<Graph>& graph);

/**
 * @brief Normalizes a graph to the unit cube and writes the inverse transformation into a transform
 *
 * @param graph The graph to normalize
 * @param transform The transform component
 */
ATCG_API void normalize(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform);

/**
 * @brief Apply a transform to a given mesh.
 * After this, the mesh vertices will be in world space and the transform will be the identity.
 *
 * @param graph The graph
 * @param transform The transform
 */
ATCG_API void applyTransform(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform);

/**
 * @brief Apply a transform to a given mesh.
 * After this, the mesh vertices will be in world space and the transform will be the identity.
 *
 * @param positions The positions
 * @param normals The normals
 * @param tangents The tangents
 * @param transform The transform
 */
ATCG_API void applyTransform(torch::Tensor& positions,
                             torch::Tensor& normals,
                             torch::Tensor& tangents,
                             atcg::TransformComponent& transform);

/**
 * @brief Convert datatype from network to host byte order.
 * @note Currently only implemented for int_t types
 *
 * @tparam T The data type
 * @param network The network representation
 * @return The host representation
 */
template<typename T>
ATCG_API T ntoh(T network);

/**
 * @brief Convert datatype from network to host byte order
 * @note Currently only implemented for int_t types
 *
 * @tparam T The data type
 * @param host The host representation
 * @return The network representation
 */
template<typename T>
T hton(T host)
{
    return ntoh(host);
}

/**
 * @brief Dump data as raw binary file to disk.
 *
 * @param path The path
 * @param data The data
 */
ATCG_API void dumpBinary(const std::string& path, const torch::Tensor& data);

/**
 * @brief Take a screenshot and save it to disk
 *
 * @param scene The scene
 * @param camera The camera
 * @param width The output width. Height is calculated from the camera's aspect ratio
 * @param path The output path
 */
ATCG_API void screenshot(const atcg::ref_ptr<Scene>& scene,
                         const atcg::ref_ptr<Camera>& camera,
                         const uint32_t width,
                         const std::string& path);

/**
 * @brief Take a screenshot and save it to disk
 *
 * @param scene The scene
 * @param camera The camera
 * @param width The output width
 * @param height The output height
 * @param path The output path
 */
ATCG_API void screenshot(const atcg::ref_ptr<Scene>& scene,
                         const atcg::ref_ptr<Camera>& camera,
                         const uint32_t width,
                         const uint32_t height,
                         const std::string& path);

/**
 * @brief Take a screenshot and return it as tensor
 *
 * @param scene The scene
 * @param camera The camera
 * @param width The output width. Height is calculated from the camera's aspect ratio
 *
 * @return The pixel data as tensor
 */
ATCG_API torch::Tensor
screenshot(const atcg::ref_ptr<Scene>& scene, const atcg::ref_ptr<Camera>& camera, const uint32_t width);

/**
 * @brief Pick an entity at the given screen coordinates. This reads information from the current framebuffer, i.e.,
 * Renderer::Framebuffer. If no entity is found, an invalid entity is returned.
 *
 * @param mouse_pos The mouse position in screen coordinates
 *
 * @return The picked entity
 */
ATCG_API Entity pickEntity(const glm::vec2& mouse_pos);

/**
 * @brief Set the sky light of the shader based on the given skybox. This will bind the irradiance map of the skybox to
 * the shader and set the according uniform. The function returns the id of the bound texture,
 *
 * @param renderer The renderer
 * @param shader The shader
 * @param skybox The skybox
 *
 * @return The id of the bound texture
 */
ATCG_API uint32_t setLights(atcg::RendererSystem* renderer,
                            Scene* scene,
                            const atcg::ref_ptr<atcg::TextureCubeArray>& point_light_depth_maps,
                            const atcg::ref_ptr<Shader>& shader);

/**
 * @brief Set the sky light of the shader based on the given skybox. This will bind the irradiance map of the skybox to
 * the shader and set the according uniform. The function returns the id of the bound texture,
 *
 * @param renderer The renderer
 * @param shader The shader
 * @param skybox The skybox
 *
 * @return The id of the bound texture
 * @return The id of the bound prefiltered map
 */
ATCG_API std::pair<uint32_t, uint32_t>
setSkyLight(atcg::RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader, const atcg::ref_ptr<Skybox>& skybox);

/**
 * @brief Display a material selection dialog and return the selected material handle. This is used in the editor and
 * returns the handle of the selected material or an invalid handle if no material was selected.
 *
 * @param key The key to identify the selection (e.g. for which component this selection is)
 * @param handle The currently selected handle (can be invalid)
 *
 * @return The handle of the selected material or an invalid handle if no material was selected
 */
ATCG_API AssetHandle displayMaterialSelection(const std::string& key, AssetHandle handle);

/**
 * @brief Display a graph selection dialog and return the selected graph handle. This is used in the editor and returns
 * the handle of the selected graph or an invalid handle if no graph was selected.
 *
 * @param key The key to identify the selection (e.g. for which component this selection is)
 * @param handle The currently selected handle (can be invalid)
 *
 * @return The handle of the selected graph or an invalid handle if no graph was selected
 */
ATCG_API AssetHandle displayGraphSelection(const std::string& key, AssetHandle handle);

/**
 * @brief Display a script selection dialog and return the selected script handle. This is used in the editor and
 * returns the handle of the selected script or an invalid handle if no script was selected.
 *
 * @param key The key to identify the selection (e.g. for which component this selection is)
 * @param handle The currently selected handle (can be invalid)
 *
 * @return The handle of the selected script or an invalid handle if no script was selected
 */
ATCG_API AssetHandle displayScriptSelection(const std::string& key, AssetHandle handle);

/**
 * @brief Display a texture selection dialog and return the selected texture handle. This is used in the editor and
 * returns the handle of the selected texture or an invalid handle if no texture was selected.
 *
 * @param key The key to identify the selection (e.g. for which component this selection is)
 * @param handle The currently selected handle (can be invalid)
 *
 * @return The handle of the selected texture or an invalid handle if no texture was selected
 */
ATCG_API AssetHandle displayShaderSelection(const std::string& key, AssetHandle handle);

/**
 * @brief Display a material selection dialog and return the selected material handle. This is used in the editor and
 * returns the handle of the selected material or an invalid handle if no material was selected.
 *
 * @param key The key to identify the selection (e.g. for which component this selection is)
 * @param handle The currently selected handle (can be invalid)
 *
 * @return The handle of the selected material or an invalid handle if no material was selected
 */
ATCG_API AssetHandle displayTexture2DSelection(const std::string& key, AssetHandle handle);

/**
 * @brief Display a texture selection dialog and return the selected texture handle. This is used in the editor and
 * returns the handle of the selected texture or an invalid handle if no texture was selected.
 *
 * @param key The key to identify the selection (e.g. for which component this selection is)
 * @param handle The currently selected handle (can be invalid)
 *
 * @return The handle of the selected texture or an invalid handle if no texture was selected
 */
ATCG_API AssetHandle displayTexture3DSelection(const std::string& key, AssetHandle handle);

/**
 * @brief Serialize a buffer
 *
 * @param file_name The file name
 * @param data The buffer data
 * @param byte_size The buffer size in bytes
 */
ATCG_API void serializeBuffer(const std::string& file_name, const char* data, const uint32_t byte_size);

/**
 * @brief Deserialize a buffer
 *
 * @param file_name The file name
 *
 * @return The deserialized data
 */
ATCG_API std::vector<uint8_t> deserializeBuffer(const std::string& file_name);

/**
 * @brief Serialize a layout
 *
 * @param layout The buffer layout
 *
 * @return The json object representing the layout
 */
ATCG_API nlohmann::json serializeLayout(const atcg::BufferLayout& layout);

/**
 * @brief Deserialize a layout
 *
 * @param layout_node The json node containing the Layout data
 *
 * @return The BufferLayout
 */
ATCG_API atcg::BufferLayout deserializeLayout(nlohmann::json& layout_node);

}    // namespace Utils

}    // namespace atcg