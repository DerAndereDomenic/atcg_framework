#pragma once

#define OPTIX_CHECK(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        OptixResult res = call;                                                                                        \
        if(res != OPTIX_SUCCESS)                                                                                       \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << "Optix error '" << optixGetErrorName(res) << "' at " << __FILE__ << ":" << __LINE__ << ": "          \
               << #call;                                                                                               \
            ATCG_ERROR(ss.str());                                                                                      \
        }                                                                                                              \
    } while(0)