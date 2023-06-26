#include <Math/Tracing.h>

namespace atcg
{

// namespace detail
// {
// float rayTriangleIntersection(const Eigen::Vector3f vertices[3],
//                               const Eigen::Vector3f& origin,
//                               const Eigen::Vector3f& direction,
//                               const float tmin,
//                               const float tmax)
// {
//     Eigen::Vector3f vertex0 = vertices[0];
//     Eigen::Vector3f vertex1 = vertices[1];
//     Eigen::Vector3f vertex2 = vertices[2];
//     Eigen::Vector3f edge1, edge2, h, s, q;
//     float a, f, u, v;
//     edge1 = vertex1 - vertex0;
//     edge2 = vertex2 - vertex0;
//     h     = direction.cross(edge2);
//     a     = edge1.dot(h);
//     if(std::abs(a) < 1e-5f) return tmax;
//     f = 1.0f / a;
//     s = origin - vertex0;
//     u = f * s.dot(h);
//     if(u < 0.0f || u > 1.0f) return tmax;
//     q = s.cross(edge1);
//     v = f * direction.dot(q);
//     if(v < 0.0f || u + v > 1.0f) return tmax;
//     float t = f * edge2.dot(q);
//     if(t > tmin) { return t; }
//     else { return tmax; }
// }

// float rayMeshIntersection(const atcg::ref_ptr<Mesh>& mesh,
//                           const Eigen::Vector3f& origin,
//                           const Eigen::Vector3f& direction,
//                           const float tmin,
//                           const float tmax)
// {
//     float best_hit = tmax;
//     for(auto f_it = mesh->faces_begin(); f_it != mesh->faces_end(); ++f_it)
//     {
//         Eigen::Vector3f vertices[3];
//         int idx = 0;
//         for(auto v_it = f_it->vertices().begin(); v_it != f_it->vertices().end(); ++v_it)
//         {
//             glm::vec3 p     = mesh->point(*v_it);
//             vertices[idx++] = Eigen::Vector3f(p.x, p.y, p.z);
//         }

//         float hit = rayTriangleIntersection(vertices, origin, direction, tmin, tmax);

//         if(hit < tmax && hit < best_hit) { best_hit = hit; }
//     }

//     return best_hit;
// }
// }    // namespace detail

// Eigen::VectorXf Tracing::rayMeshIntersection(const atcg::ref_ptr<Mesh>& mesh,
//                                              const Eigen::MatrixX3f& origin,
//                                              const Eigen::MatrixX3f& direction,
//                                              const float tmin,
//                                              const float tmax)
// {
//     size_t num_paths = origin.rows();

//     Eigen::VectorXf result(num_paths);

//     for(uint32_t i = 0; i < num_paths; ++i)
//     {
//         result[i] = detail::rayMeshIntersection(mesh, origin.row(i), direction.row(i), tmin, tmax);
//     }

//     return result;
// }

}    // namespace atcg