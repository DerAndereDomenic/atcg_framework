# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

C++20 research rendering framework (`atcg_lib`) combining OpenGL rasterization, CUDA/OptiX path tracing, and libtorch tensors. It also ships Python bindings (`pyatcg`). It is developed mainly on Windows with MSVC; gcc is used on Linux. Dependencies are git submodules in `external/`. The exceptions are torch, Python and CUDA, which must be installed on the system. Run `git submodule update --init --recursive` after cloning.

## Build

```
cmake . -B build                                   # configure (CUDA backend ON by default)
cmake --build build --parallel --config <Debug|Release|RelWithDebInfo>
cmake --build build --config Release --target <TargetName>   # e.g. Sandbox, PBR, SimpleIntegrator
```

CMake options:
- `ATCG_CUDA_BACKEND`: default ON. Turning it off removes all of OptiX/integrators/BSDFs. CI builds with `-DATCG_CUDA_BACKEND=Off` on Ubuntu and Windows, so keep CPU-only builds compiling.
- `ATCG_HEADLESS`: EGL with no GLFW/ImGui. Linux and CUDA only.
- `ATCG_BUILD_TESTS`, `ATCG_BUILD_DOCS`, `ATCG_BUILD_EXAMPLES` (default ON), `ATCG_SPECTRAL_RENDERING`, `ATCG_PYTHON_MODULE` (set by the pip build).

Torch is found through the active Python installation, using charonload. On Windows, `<python>/Lib/site-packages/torch/lib` must be on `PATH` at runtime.

All executables, DLLs and `.ptx` files are written to `bin/`, and libraries to `lib/`. Executables load PTX through relative paths such as `./bin/MeshShape_ptx.ptx`. They also expect shaders and resources relative to the repo root. **Run binaries from the repository root.** (The MSVC debugger working directory is already set to the root.)

## Tests

GoogleTest. Configure with `-DATCG_BUILD_TESTS=On` and build the `atcg_lib_test` target. Then, from the repo root:

```
./bin/<Config>/atcg_lib_test                         # or ./bin/atcg_lib_test depending on generator
./bin/<Config>/atcg_lib_test --gtest_filter=Scene*   # single suite/test
ctest --test-dir build -C <Config> -R <pattern>
```

Test sources are `tests/*_test.cpp`, collected by glob. Test data is in `tests/data`.

## Python bindings

`pip install -e .` builds `pyatcg` JIT via charonload on first `import pyatcg`. The package lives in `atcg_lib/platform/python/pyatcg`, and the bindings are in `atcg_lib/platform/python/pyatcg.cpp`. Environment variables set before import configure the build: `ATCG_CUDA_BACKEND`, `ATCG_HEADLESS`, `ATCG_BUILD_TYPE` (default RelWithDebInfo), `ATCG_STUBS_DIR`, `ATCG_VERBOSE`. Examples are `PythonFrontend.py` and `PythonHeadless.py`. A normal C++ build also embeds Python (`ATCG_INIT_PYTHON`) for scripting; see `src/PBR`.

## Formatting

`.clang-format` uses Allman braces, a 120-column limit, 4-space indent, and aligned consecutive assignments and macros. Format C++/CUDA changes with clang-format.

## Architecture

### Library source layout (`atcg_lib/`)
- `include/` + `src/`: platform-independent core, split by area: Core, Renderer, Scene, Asset, DataStructure, Events, Math, Network, Plugin, Scripting.
- `platform/<x>/` holds backend-specific code. CMake picks it up by globbing the selected backend:
  - `opengl/`: implementation of the Renderer abstractions. Always compiled.
  - `cuda/` or `cpu/`: compute backend chosen by `ATCG_CUDA_BACKEND`.
  - `glfw/` or `headless/`: window/context chosen by `ATCG_HEADLESS`.
  - `optix/`: the whole ray tracing stack. Only built with CUDA. Its headers are included as top-level paths such as `<BSDF/...>`, `<Integrator/...>`, `<Shape/...>`.
  - `python/`: pybind/charonload bindings.
- Platform include dirs are added to the public include path. Guard code that is optional with `#ifdef ATCG_CUDA_BACKEND` and `#ifndef ATCG_HEADLESS`. ImGui code must be inside the latter guard.
- Sources are found by `GLOB_RECURSE`, so re-run CMake configure after adding or removing files.

### Application model
The design is Hazel-style. An app defines `atcg::createApplication()` (entry point in `Core/EntryPoint.h`), subclasses `atcg::Application`, and pushes `atcg::Layer`s. Layers implement `onAttach`, `onUpdate`, `onImGuiRender` and `onEvent`. Engine subsystems (Renderer, ShaderManager, logger, etc.) are registered in the `SystemRegistry` singleton. The registry is handed to dynamically loaded DLLs through `setInstance`, so plugins share engine state. Scenes use EnTT (`Scene`, `Entity`, `Scene/Components`). `ComponentRegistry` handles component serialization, GUI and rendering registration. Memory is managed with `atcg::ref_ptr`/`make_ref`, which are `std::shared_ptr`. For CUDA device memory, use `atcg::dref_ptr`/`DeviceBuffer` and `upload()` from a host object; never use `make_ref` for device objects. GPU buffers and textures expose torch tensors. Mapped OpenGL buffers must be unmapped before OpenGL uses them again.

### Executables and plugins
- `src/<Name>/`: each folder with a `CMakeLists.txt` is auto-added as an executable named after the folder, via `ATCG_add_executable` in `cmake/ATCGFunctions.cmake`. `ATCG_TARGET_DIR` points to the folder, so apps can load their local shaders.
- `plugins/<Name>/`: shared libraries built with `ATCG_add_plugin`. They are aggregated by the `AllPlugins` target and loaded at runtime by `PluginManagerSystem`. A plugin uses `ATCG_PLUGIN_LIBRARY();` and exports `extern "C" registerPlugin(atcg::PluginRegistry&)` (plus `registerPythonBindings`). In `registerPlugin` it registers materials, render passes, components, integrators or BSDFs by string type name.

### OptiX / CUDA device code
- Device programs are `*_ptx.cu` files whose first line is `#pragma cuda_source_property_format = PTX`. `cmake/FilterPTX.cmake` finds them by that pragma. It compiles them into a `<prefix>_ptxmodules` object library and copies the `.ptx` into `bin/`. Without the pragma a `.cu` file is compiled as normal CUDA into the host target.
- Host classes (`Foo.h/.cpp`) build an OptiX pipeline and SBT. They reference PTX entry points by file and name, e.g. `{"./bin/Foo_ptx.ptx", "__raygen__rg"}`. Launch parameters shared between host and device go in `FooData.cuh`.
- Polymorphism on the device uses "VPtrTable" structs (`*VPtrTable.cuh`). These hold OptiX direct-callable indices. BSDFs, emitters, sensors, films, media, phase functions and shape samplers all use them. Integrators build on `Integrator`, `SceneAdapter` (Scene → OptiX scene/IAS) and `RaytracingPipeline`/`ShaderBindingTable`. `plugins/SimpleIntegrator` is a minimal reference implementation.
- Radiance is carried as `SampledSpectrum`/`SampledWavelengths`. `ATCG_SPECTRAL_RENDERING` decides whether the renderer is truly spectral.

### Shaders and resources
OpenGL shaders are in `shader/`: `<name>.vs/.fs[/.gs]`, compute `.glsl`. They are loaded with `atcg::ShaderManager::addShaderFromName("<name>")` and hot-reload at runtime. Meshes and textures are in `res/`. The directories are available as compile-time macros `ATCG_SHADER_DIR`, `ATCG_RESOURCE_DIR` and `ATCG_BASE_DIR`.

### Subsystem guides
These load automatically when working in the matching directories:
- Rasterization renderer (render graph, passes, materials, shaders): [atcg_lib/src/Renderer/CLAUDE.md](atcg_lib/src/Renderer/CLAUDE.md). It is also imported from `atcg_lib/include/Renderer/`, `atcg_lib/platform/opengl/` and `shader/`.
- OptiX path tracer (pipeline/SBT, VPtrTables, SceneAdapter, integrators): [atcg_lib/platform/optix/CLAUDE.md](atcg_lib/platform/optix/CLAUDE.md)

## Changelog

`CHANGELOG.md` follows Keep a Changelog. Add user-facing changes under `[Unreleased]`.

### Rules

## General
- Do not compile the project yourself.
- Do not run any binaries yourself.
- Always make sure the formatting is applied
- Do not commit any code or push code
- Do not compile or run the tests unless I tell you to

## Coding rules
- Adhere to the coding conventions when defining new classes, variables and functions
- Never use try-catch blocks