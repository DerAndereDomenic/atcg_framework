#pragma once

// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
#define ATCG_BIND_EVENT_FN(fn)                                                                                         \
    [this](auto&&... args) -> decltype(auto)                                                                           \
    {                                                                                                                  \
        return this->fn(std::forward<decltype(args)>(args)...);                                                        \
    }
