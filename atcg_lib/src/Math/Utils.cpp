#include <Math/Utils.h>

#include <fstream>

namespace atcg
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

namespace IO
{
void dumpBinary(const std::string& path, const torch::Tensor& data)
{
    auto data_ = data.to(atcg::CPU);
    std::ofstream out(path, std::ios::out | std::ios::binary);
    out.write((const char*)data_.data_ptr(), data_.numel() * data_.element_size());
}
}    // namespace IO
}    // namespace atcg