#pragma once

#include <optix_types.h>
#ifdef ATCG_CUDA_BACKEND
    #include <optix.h>

    #ifdef NDEBUG
        #define OPTIX_CHECK(call) call
    #else
        #ifndef __CUDACC__
            #define OPTIX_CHECK(call)                                                                                  \
                do                                                                                                     \
                {                                                                                                      \
                    OptixResult res = call;                                                                            \
                    if(res != OPTIX_SUCCESS)                                                                           \
                    {                                                                                                  \
                        std::stringstream ss;                                                                          \
                        ss << "Optix error '" << optixGetErrorName(res) << "' at " << __FILE__ << ":" << __LINE__      \
                           << ": " << #call;                                                                           \
                        ATCG_ERROR(ss.str());                                                                          \
                    }                                                                                                  \
                } while(0)
        #endif
    #endif
#else
    #define OPTIX_CHECK(call) call
#endif