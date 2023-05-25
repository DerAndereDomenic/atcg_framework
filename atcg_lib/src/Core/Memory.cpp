#include <Core/Memory.h>

namespace atcg
{
std::size_t host_allocator::bytes_allocated   = 0;
std::size_t host_allocator::bytes_deallocated = 0;

#ifdef ATCG_CUDA_BACKEND
std::size_t device_allocator::bytes_allocated   = 0;
std::size_t device_allocator::bytes_deallocated = 0;
#endif
}    // namespace atcg