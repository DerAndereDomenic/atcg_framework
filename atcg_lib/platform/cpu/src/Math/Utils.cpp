#include <Math/Utils.h>

namespace atcg
{

void normalize(const atcg::ref_ptr<Graph>& graph)
{
    float max_scale = -std::numeric_limits<float>::infinity();
    glm::vec3 mean_point(0);

    Vertex* vertices = (Vertex*)graph->getVerticesBuffer()->getHostPointer();

    for(uint32_t i = 0; i < graph->n_vertices(); ++i)
    {
        max_scale = glm::max(max_scale, vertices[i].position.x);
        max_scale = glm::max(max_scale, vertices[i].position.y);
        max_scale = glm::max(max_scale, vertices[i].position.z);

        mean_point += vertices[i].position;
    }
    mean_point /= static_cast<float>(graph->n_vertices());

    for(uint32_t i = 0; i < graph->n_vertices(); ++i)
    {
        vertices[i].position = (vertices[i].position - mean_point) / max_scale;
    }

    graph->getVerticesBuffer()->unmapPointers();
}
}    // namespace atcg