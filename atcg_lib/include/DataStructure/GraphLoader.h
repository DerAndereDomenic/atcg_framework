#pragma once

namespace atcg
{

class Graph;
class Scene;

namespace IO
{

/**
 * @brief This function tries to figure out what kind of geometry should be loaded.
 * If faces are present, it assumes a triangle mesh (other tesselations are not supported).
 * If line are present, it assumes a line collection.
 * If points are present, it assumes a point cloud. Additionally, if no shape information is given (neither f, l, or p),
 * it is assumed to be a point cloud file and all vertices are added to the geometry.
 *
 * @param path The path to the obj file
 *
 * @return The geometry representation
 */
atcg::ref_ptr<Graph> read_any(const std::string& path);

/**
 * @brief Read an obj (triangle) mesh.
 * If the obj file contains multiple objects, they are all combined into one mesh representation.
 * Material files are ignored
 *
 * @param path The path to the obj file
 *
 * @return The mesh representation
 */
atcg::ref_ptr<Graph> read_mesh(const std::string& path);

/**
 * @brief Read an obj triangle cloud.
 * This method expects shapes that are created using the obj p tag. Face information is ignored.
 * If the obj file contains multiple objects, they are all combined into one point cloud representation.
 * Material files are ignored.
 *
 * @param path The path to the obj file
 *
 * @return The point cloud representation
 */
atcg::ref_ptr<Graph> read_pointcloud(const std::string& path);

/**
 * @brief Read an obj file representing a line collection.
 * This method ignores all surface models (triangle meshes) and only creates geometry from the line tag (l).
 * Only line segments with two vertices are supported (i.e. l x y).
 * Material files are ignored
 *
 * @param path The path to the obj file
 *
 * @return The line representation
 */
atcg::ref_ptr<Graph> read_lines(const std::string& path);

/**
 * @brief Read a scene representation from an obj file.
 * Create a scene object where each object is either a triangle mesh (f tag), a point cloud (p tag) or a line
 * collection. To use material files, export the obj with the pbr extension. The following parameters are used:
 *
 * - Kd - diffuse base color
 * - Pr - surface roughness
 * - Pm - surface metallic
 * - map_Kd - diffuse base color texture
 * - map_Pr - roughness texture
 * - map_Pm - metallic texture
 * - map_Bump - normal texture
 *
 * @note We do not use per-face materials but only per object materials
 *
 * @param path The path to the obj file
 *
 * @return The line representation
 */
atcg::ref_ptr<Scene> read_scene(const std::string& path);
}    // namespace IO
}    // namespace atcg