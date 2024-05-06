#include "kernels.h"
#ifndef ATCG_CUDA_BACKEND


void simulate(atcg::BufferView<glm::vec3>& points, uint32_t size, float time)
{
    for(size_t tid = 0; tid < size; ++tid)
    {
        int j = tid % size;
        int i = tid / size;

        updatePoint(points, tid, j, i, time);
    }
}

#endif