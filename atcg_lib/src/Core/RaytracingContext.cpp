#include <Renderer/RaytracingContext.h>

#include <Core/Optix.h>
#include <Core/CUDA.h>
#include <Core/Assert.h>

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

namespace atcg
{
namespace detail
{
static bool s_optix_initialized = false;
}

void RaytracingContext::initRaytracingAPI()
{
    if(!detail::s_optix_initialized)
    {
#ifdef ATCG_CUDA_BACKEND
        OPTIX_CHECK(optixInit());
#endif
        detail::s_optix_initialized = true;
    }
}

void RaytracingContext::destroy()
{
#ifdef ATCG_CUDA_BACKEND
    ATCG_ASSERT(_context != nullptr, "Try to destroy context before creation");

    optixDeviceContextDestroy(_context);
    _context = nullptr;
#endif
}

void RaytracingContext::create(const int device_id)
{
    ATCG_ASSERT(_context == nullptr, "Try to create context while already defined");

#ifdef ATCG_CUDA_BACKEND
    int original_device = 0;
    CUDA_SAFE_CALL(cudaGetDevice(&original_device));

    CUDA_SAFE_CALL(cudaSetDevice(device_id));

    OptixDeviceContextOptions options = {};

    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &_context));

    CUDA_SAFE_CALL(cudaSetDevice(original_device));
#endif
}
}    // namespace atcg
