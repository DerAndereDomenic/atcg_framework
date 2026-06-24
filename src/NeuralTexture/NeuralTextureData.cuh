#pragma once

#include <Neural/DeviceMLP.cuh>
#include <Neural/DeviceHashGrid.h>

struct NeuralTextureData
{
    int width;
    int height;
    glm::vec4* output;
    glm::vec4* grad_output;

    atcg::DeviceMLP<3, 32, 64, 8>* device_mlp;
    atcg::DeviceHashGrid<half, 16, 2>* device_hash_grid;
};