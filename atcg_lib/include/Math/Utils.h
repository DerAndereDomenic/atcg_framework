#pragma once

#include <memory>
#include <numeric>

namespace atcg
{
    /**
     * @brief Normalize a mesh to be in the unit cube
     */
    template<class Geometry>
    void normalize(const std::shared_ptr<Geometry>& geom)
    {
        double max_scale = -std::numeric_limits<double>::infinity();
        typename Geometry::Point mean_point{0,0,0};
        for(auto v_it = geom->vertices_begin(); v_it != geom->vertices_end(); ++v_it)
        {
            for(uint32_t i = 0; i < 3; ++i)
            {
                if(geom->point(*v_it)[i] > max_scale)
                {
                    max_scale = geom->point(*v_it)[i];
                }
            }

            mean_point += geom->point(*v_it);
        }
        mean_point /= static_cast<double>(geom->n_vertices());

        for(auto v_it = geom->vertices_begin(); v_it != geom->vertices_end(); ++v_it)
        {
            geom->set_point(*v_it, (geom->point(*v_it) - mean_point) / max_scale);
        }
    }
}