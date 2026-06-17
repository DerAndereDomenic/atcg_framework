#pragma once

#include <Neural/DeviceMLP.cuh>

struct NeuralTextureData
{
    int width;
    int height;
    glm::vec4* output;
    glm::vec4* grad_output;

    atcg::DeviceMLP<3, 8, 64, 8>* device_mlp;
};