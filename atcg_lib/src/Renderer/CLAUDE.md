# OpenGL rasterization renderer

This file covers the rasterization renderer. Its code is spread over four places:
- `atcg_lib/include/Renderer/` + `atcg_lib/src/Renderer/`: API-agnostic abstractions, render graph, passes, materials, cameras.
- `atcg_lib/platform/opengl/src/Renderer/`: the only place that calls OpenGL (`Buffer`, `Texture`, `Framebuffer`, `Shader`, `ShaderCompiler`, `VertexArray`, `GraphicsAPI`, `VRSystem`).
- `atcg_lib/src/Scene/SceneRenderer.cpp`, `Scene/Components/*`: turn a `Scene` into draw calls.
- `shader/`: GLSL sources.

## Rules

- Raw `gl*` calls belong only in `platform/opengl`. Elsewhere, use the abstractions: `GraphicsCommand::*` (`Renderer/GraphicsAPI.h`), `Texture*`, `Framebuffer`, `VertexBuffer`/`VertexArray`, `Shader`.
- Pipeline state is set through `GraphicsPipeline` (shader + `RasterizerState` + `PrimitiveTopology`), which is passed to `drawVAO`. Don't toggle GL state by hand.
- Texture units come from a pool. Always pair `renderer->popTextureID()` with `renderer->pushTextureID(id)` after drawing, or units leak.
- The renderer, shader manager and scene renderer are systems in `SystemRegistry`. The free-function facades (`atcg::Renderer::`, `atcg::ShaderManager::`, `atcg::SceneRenderer::`) forward to them. Inside passes and component renderers, use the `RendererSystem*` you are given (`ctx.renderer` / `renderer`), not the global facades.
- OpenGL buffers and textures can be mapped as torch tensors or CUDA pointers. Unmap them (`unmapDevicePointers()` / `unmap*Pointer()`) before OpenGL touches them again.
- Headless builds (`ATCG_HEADLESS`) have no GLFW or ImGui. Keep ImGui code inside `#ifndef ATCG_HEADLESS`.

## Frame flow

`Application` calls each `Layer::onUpdate`. User code calls `atcg::SceneRenderer::render(scene, camera, target_fbo)`, which runs the scene renderer's `RenderGraph` with a `RenderContext {renderer, scene, camera, draw_cameras}`.

### Render graph (`RenderGraph.h/.cpp`)
- Nodes are `RenderPass` subclasses. A pass implements:
  - `reflect(CompileData)`: declares named inputs (`addInput`) and outputs (`addOutput(name, ResourceDescription)`). It can also say which outputs form its framebuffer (`setOutputFramebufferData(width, height, {handles})`), using `TextureSizeHint` such as `FULL_FRAMEBUFFER`.
  - `execute(RenderContext, ResourceTable)`: reads inputs with `resources.getTexture<T>(name)`, `getTensor` or `getBuffer`. It renders into `resources.getTargetFBO()` between `GraphicsCommand::beginRenderPass/endRenderPass`.
- Edges are made with `graph->addDependency(src_handle, "src_output", dst_handle, "dst_input")`. Port names are plain strings and are **not validated**. An unknown source port is logged as an error each time resources are allocated (once per new output resolution). An unknown target port is ignored without any message. Check port names against the passes' `reflect()`.
- `compile()` sorts the passes topologically and reflects them. Physical resources are allocated lazily for each output resolution. Up to `_max_cached_resolutions` resolutions are cached. Changing pass structure or MSAA sample count requires recompiling.
- Physical resources are keyed by `pass->name() + "." + port`. Pass names are type names (e.g. `"BlitPass"`), so **two instances of the same pass type in one graph collide** and share or overwrite each other's resources. Don't add a second instance of a pass type until this is fixed.
- Every graph ends in the built-in `OutputPass`. It takes `color`, `entities`, `stencil` and `depth`, and writes them to the target FBO.
- `SceneRendererSystem::Impl::createRenderGraph()` (`Scene/SceneRenderer.cpp`) builds the default graph:
  `DepthPass` + `ShadowPass` → `ForwardPass` → (`BlitPass` if MSAA) → `TonemapPass` → `OutlinePass` → `OutputPass`.
  Users can replace it with `SceneRenderer::setRenderGraph`.

### Adding a render pass
1. Subclass `RenderPass`. Its constructor must take `Dictionary& properties`, which is used for configuration.
2. Add a static `registerRenderPass(RenderPassRegistry::Registry*)` that uses `ATCG_REGISTER_RENDER_PASS(registry, "Name", Class)`.
3. Built-in passes are registered in `Core/SystemCollection.cpp`. Plugins call `registry.registerRenderPass<T>("Name")`.
4. Create passes through `RenderPassRegistry::createRenderPass("Name", dict)`.
Existing passes in `src/Renderer/RenderPasses/` are the templates to follow. `ForwardPass` is the most complete.

### Drawing scene components
`ForwardPass` goes through every entity with a `TransformComponent`. It draws opaque entities first, then transparent ones (`TransparencyComponent`) sorted back to front. Each entity is drawn with `ComponentRegistry::renderAllComponents(renderer, entity, camera, auxiliary)`. Each render component (`MeshRenderComponent`, `PointSphereRenderComponent`, `EdgeCylinderRenderComponent`, `InstanceRenderComponent`, …) has a `ComponentRenderer<T>` specialization. The `auxiliary` `Dictionary` carries `point_light_depth_maps`, `skybox`, `has_skybox`, `depth_map`, `draw_cameras`, and optionally `override_shader`. A new renderable component needs a `ComponentRenderer<T>` specialization and registration via `ATCG_REGISTER_COMPONENT` (or `registerComponent<T>` in a plugin).

### Materials
`Material` (`Renderer/Material.h`) is an asset with `uploadMaterial(renderer, shader)` and `clone()`. The concrete types are `OpaqueMaterial` (PBR textures for diffuse, metallic and roughness), `DielectricMaterial` and `NullMaterial`. `getMaterialType()` returns a string. The OptiX backend uses the same string to choose a BSDF, so a new material type usually needs a matching BSDF as well (see `atcg_lib/platform/optix/CLAUDE.md`).

## Shaders (`shader/`)
- A program is made of `<name>.vs` + `<name>.fs` [+ `<name>.gs`]. A compute shader is a single `<name>.glsl`. Load them with `ShaderManager::addShaderFromName("<name>")` / `addComputeShaderFromName`, then use `getShader("<name>")`. Built-in shaders are loaded by the framework. An app's own shaders live in its `src/<App>/` folder (`ATCG_TARGET_DIR`).
- `ShaderCompiler` handles `#include "relative/path.glsl"` itself, relative to the including file. It also collects `subroutine uniform` declarations. Shared GLSL is in `shader/common/` (`defines.glsl`, `bsdf.glsl`, `bsdf_functions.glsl`, `ibl.glsl`).
- `ShaderManager::onUpdate()` watches file timestamps and recompiles at runtime. Shader edits don't need a C++ rebuild, but compile errors only appear in the log at runtime.
- `shader/common/bsdf_functions.glsl` and the OptiX `BSDF/BSDFFunctions.h` implement the same GGX/Schlick microfacet model. Keep them consistent so the raster preview matches the path-traced result.
