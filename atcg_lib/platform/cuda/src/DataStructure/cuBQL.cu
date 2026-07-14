#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <DataStructure/cuBQL.h>

#include <cuBQL/bvh.h>
#include <cuBQL/builder/cuda.h>

#include <c10/cuda/CUDACachingAllocator.h>

namespace atcg
{

namespace detail
{
struct TorchMemoryResource : public cuBQL::GpuMemoryResource
{
    void malloc(void **ptr, size_t size, cudaStream_t s) override
    {
        // Use torch cache allocator
        *ptr = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, s);
    }

    void free(void *ptr, cudaStream_t s) override { c10::cuda::CUDACachingAllocator::raw_delete(ptr); }
};

TorchMemoryResource &defaultTorchMemResource()
{
    static TorchMemoryResource memResource;
    return memResource;
}

}    // namespace detail


void build3fBVH(cuBQL::BinaryBVH<float, 3> &bvh,
                /*! array of bounding boxes to build BVH over, must
                  be in device memory */
                const cuBQL::box_t<float, 3> *boxes,
                uint32_t numBoxes)
{
    cuBQL::gpuBuilder(bvh, boxes, numBoxes, cuBQL::BuildConfig(), 0, detail::defaultTorchMemResource());
}

void free3fBVH(cuBQL::BinaryBVH<float, 3> &bvh)
{
    cuBQL::free(bvh, 0, detail::defaultTorchMemResource());
}
}    // namespace atcg