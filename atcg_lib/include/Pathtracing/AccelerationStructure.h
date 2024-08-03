#pragma once

#include <torch/types.h>
#include <DataStructure/Graph.h>
#include <nanort.h>

namespace atcg
{
class AccelerationStructure
{
public:
    AccelerationStructure() = default;

    virtual ~AccelerationStructure() {}

    inline torch::Tensor getPositions() const { return _positions; }
    inline torch::Tensor getNormals() const { return _normals; }
    inline torch::Tensor getUVs() const { return _uvs; }
    inline torch::Tensor getFaces() const { return _faces; }

protected:
    torch::Tensor _positions;
    torch::Tensor _normals;
    torch::Tensor _uvs;
    torch::Tensor _faces;
};

class BVHAccelerationStructure : public AccelerationStructure
{
public:
    BVHAccelerationStructure() = default;

    BVHAccelerationStructure(const atcg::ref_ptr<Graph>& graph);

    ~BVHAccelerationStructure();

    inline const nanort::BVHAccel<float>& getBVH() const { return _bvh; }

    inline void setBVH(const nanort::BVHAccel<float>& bvh) { _bvh = bvh; }

private:
    nanort::BVHAccel<float> _bvh;
};

}    // namespace atcg