# OptiX ray tracing backend

Everything under `atcg_lib/platform/optix/` is compiled only when `ATCG_CUDA_BACKEND` is ON. Its `include/` folder is on the public include path, so headers are included as `<BSDF/...>`, `<Integrator/...>`, `<Core/RaytracingPipeline.h>`, etc. Some of these names are the same as headers in `atcg_lib/include/Core/`.

## Rules

- Code outside this directory that uses OptiX types must be inside `#ifdef ATCG_CUDA_BACKEND`. CI builds with CUDA off.
- Headers shared by host and device must compile under both `nvcc` (PTX) and the host compiler. Use `ATCG_HOST_DEVICE`, and put device-only members inside `#ifdef __CUDACC__`. Host-only includes (ImGui, `ComponentGUIHandler`) go inside `#ifndef __CUDACC__`.
- PTX modules are compiled separately from `atcg_lib`. They inherit its include dirs and compile definitions but do not link against it. Device code can only use header-only code, `__device__` functions and OptiX intrinsics.
- PTX entry-point names (`__raygen__*`, `__miss__*`, `__closesthit__*`, `__direct_callable__*`) and PTX file paths are plain strings in host code. If you rename one side, grep for and update the other. By convention, direct-callable names carry a type suffix, e.g. `__direct_callable__sample_pbrbsdf`.
- After changing a `*_ptx.cu` file, rebuild the owning target so the `-copy` step refreshes `bin/*.ptx`. The binary loads `./bin/*.ptx` at runtime, so a stale PTX file causes confusing behavior.
- Device-side objects are allocated with `atcg::dref_ptr<T>` / `atcg::DeviceBuffer<T>` and filled with `.upload(&host_obj)`. Never use `make_ref` for device memory.

## File conventions (per component `Foo`)

| File | Role |
|---|---|
| `include/<Area>/Foo.h`, `src/<Area>/Foo.cpp` | Host class. Builds data, registers programs in `initializePipeline`, and provides ImGui. |
| `include/<Area>/FooData.cuh` | POD struct shared by host and device (SBT record data or launch params). |
| `src/<Area>/Foo_ptx.cu` | Device programs. **The first line must be `#pragma cuda_source_property_format = PTX`**. That pragma is how `cmake/FilterPTX.cmake` finds the file, and it produces `bin/Foo_ptx.ptx`. |
| `src/<Area>/Foo.cu` (no pragma) | Ordinary CUDA kernels compiled into `atcg_lib` (e.g. `HDRFilm.cu`, `MeshKernels.cu`). |

## Core objects (`Core/`)

- `RaytracingContext`: the OptiX device context.
- `RayTracingPipeline`: `addRaygenShader`, `addMissShader`, `addCallableShader`, `addTrianglesHitGroupShader(shape_type, slot, closesthit, anyhit)`, then `createPipeline()`. Each takes a `ShaderEntryPointDesc {ptx_filename, entrypoint}`. `numRays` is the number of ray types. `getRay(...)` returns `TraceParameters`, which device code passes to `traceWithDataPointer`.
- `ShaderBindingTable`: `add{Raygen,Miss,Hit,Callable}Entry(prog_group[, record_data])` returns the SBT index, then `createSBT()`. Record data is copied byte for byte. Usually it is a device pointer to the component's `FooData`.
- `OptixComponent`: base for BSDFs, emitters, sensors, shapes, media and so on. It has `initializePipeline(pipeline, sbt)` and `ensureInitialized`. A component adds its programs to the integrator's pipeline and SBT exactly once.

## Device polymorphism: VPtrTables

There are no virtual functions on the device. Each polymorphic family (BSDF, Emitter, Sensor, Film, Medium, PhaseFunction, ShapeSampler) has a `*VPtrTable` struct in `*VPtrTable.cuh`. The struct holds SBT callable indices plus flags. Its `__device__` methods dispatch with `optixDirectCall`.

The pattern, using `BSDF/PBRBSDF.cpp` as the reference:
1. In `initializePipeline`, call `pipeline->addCallableShader({"./bin/Foo_ptx.ptx", "__direct_callable__..."})` for each method.
2. Call `sbt->addCallableEntry(group, _data_buffer.get())` and keep the returned index.
3. Fill a host `FooVPtrTable` with the indices and flags, then `_vptr_table.upload(&table)`.
4. Integrators and launch params store `const FooVPtrTable*` device pointers and call e.g. `bsdf->sampleBSDF(si, wavelengths, rng, ctx)`.

If you add a method to a family, update all of these: the VPtrTable struct (index field + device method), the registration in every implementation's `initializePipeline`, and every `_ptx.cu` of that family.

## Scene → OptiX (`Scene/SceneAdapter.cpp`)

`SceneAdapter(context, pipeline, sbt).apply(scene, width, height)` turns an `atcg::Scene` into an `OptixScene`. The `OptixScene` holds shape instances, the IAS, emitters, an environment emitter and a sensor/film. `apply` handles these render components through `prepareComponent<T>`: `MeshRenderComponent`, `PointSphereRenderComponent`, `EdgeCylinderRenderComponent`, `InstanceRenderComponent` and `MeshLightComponent`. Materials become BSDFs through `BSDFRegistry::createBSDF(material->getMaterialType(), ...)`. The **BSDF registry key must equal the rasterizer material type string** (e.g. `"Opaque"` → `PBRBSDF`). Shapes and BSDFs are cached per `AssetHandle`. To support a new renderable component in path tracing, add a `prepareComponent<T>` specialization and call it in `apply`.

## Integrators (`Integrator/`)

- Subclass `Integrator(context, dict)`. It provides `_pipeline` and `_sbt`.
  - In the constructor, register the shape hit groups (e.g. `MeshShape` → `"./bin/MeshShape_ptx.ptx"`, `__closesthit__mesh`), run `SceneAdapter`, add your raygen and miss programs, then call `createPipeline()` + `createSBT()`. The scene arrives as `dict["scene"]`, with `dict["width"]` and `dict["height"]`.
  - Implement `generateRays(Dictionary&)`, `reset()` and `onImGuiRender()`.
- `generateRays` fills a `*Params` struct (from `FooData.cuh`), uploads it, and calls `_pipeline->launch(...)`. It writes `"output"` (radiance tensor) and `"entity_ids"` into the dictionary. Consumers (`src/PBR`, the Python bindings) read those keys.
- Built-ins (`PathtracingIntegrator`, `VolPathtracingIntegrator`, `PhotonMapIntegrator`), `BSDF`s and other components are registered in `atcg_lib/src/Core/SystemCollection.cpp`. Plugins register through `registry.registerIntegrator<T>("Name")` / `registerBSDF`. They are created by name with `IntegratorRegistry::createIntegrator(name, context, dict)`.
- `plugins/SimpleIntegrator` (BSDF-only / emitter-only sampling) is the smallest complete example. Copy it when you start a new integrator.

## Rendering math conventions

- Radiance is `SampledSpectrum` with `SampledWavelengths` sampled in [380, 780] nm. With `ATCG_SPECTRAL_RENDERING` off it is effectively RGB, so write code that works in both modes.
- Track `TransportMode` (radiance vs. importance) in the `BSDFContext`, and track the relative IOR `eta` from BSDF sampling. Russian roulette weights depend on it.
- MIS helpers are in `Integrator/MIS.h`. Ray/frame utilities are in `DataStructure/Ray.h` and `Frame.h`. RNG is `atcg::PCG32`, seeded with `sampleTEA64(pixel, frame)` (`Math/Random.h`).
- `Neural/` contains device MLP and hash-grid encodings (cooperative vectors) used by the NRC and NeuralTexture examples.
