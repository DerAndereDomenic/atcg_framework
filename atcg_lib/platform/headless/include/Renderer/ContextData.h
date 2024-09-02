#pragma once

#include <EGL/egl.h>

namespace atcg
{
struct ContextData
{
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
    int device_id;
};
}    // namespace atcg