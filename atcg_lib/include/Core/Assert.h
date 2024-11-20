#pragma once

#include <Core/Log.h>

#ifdef NDEBUG

    #define ATCG_ASSERT(condition, message)

#else

    #define ATCG_ASSERT(condition, message)                                                                            \
        if(!(condition))                                                                                               \
        {                                                                                                              \
            ATCG_ERROR("Assertion failed: {0} in {1}:{2} \n {3}", #condition, __FILE__, __LINE__, message);            \
            abort();                                                                                                   \
        }

#endif