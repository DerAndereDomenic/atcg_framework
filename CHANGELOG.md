# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Note that this project is currently in a beta state. Therefore, there might be API changes that are not reflected in the versioning.

## [Unreleased]

### Added

- Added Cubemap Array textures
- Possibility to create textures from device tensors
- Added function to remove entities from a scene
- Added possibility to remove entities via the scene hierarchy panel
- Added Python bindings for Entity Management in the scene class
- Added Point lights
- Added Shadow mapping for point light on and from meshes (other draw types currently not supported)
- Added function to draw singular components of an entity
- Added possibility to override the shaders used by an entitie's component
- Added possibility to override the shaders of RendererSystem::draw on meshes
- Added entity.entity_handle() method
- Added Performance Panel to monitor frame time
- Added multi sampled textures
- Added multi sampled framebuffers
- Added multi sample anti aliasing (MSAA)

### Changed

- The Renderer now internally also uses the Texture ID push system
- Cube maps are no longer interpreted as depth 6 textures
- Entity lookup in a scene by name is now approximately O(1) via hashing
- If an Entity can't be found by ID, an invalid entity is returned
- Circle shader now also writes out entity IDs
- Removed RendererSystem::draw function that takes in a material struct
- At the end of each frame, all texture units are unbinded to create a clean state for the next frame
- It is now possible to attach arbitrary textures (2D, 3D, Cube, etc.) to framebuffers. This changed the return value of fbo->getColorAttachament() from atcg::ref_ptr<Texture2D> to atcg::ref_ptr<Texture>
- fbo->blit now explicitely tries to copy all color attachements
- RendererSystem::useScreenBuffer now uses the MSAA framebuffer if MSAA is enabled (default). If MSAA is turned off, the old framebuffer and behavior is used
- RendererSystem::getFramebuffer now returns the defacto state of the last frame. Direct render calls to this framebuffer will have no errect if MSAA is enabled because it will get overwritten by the blitting of the framebuffers. Use RendererSystem::getFramebufferMSAA() instead. This behavior is also the case for getFrame and getZBuffer

### Fixed

- Screenshot functionality of Renderer uses correct member method and not global renderer instance.
- scene->removeAllEntities now clear internal buffers
- Fixed Texture tests on CPU backend
- Fix normals of cylinder in test scene
- Normals of cylinder in test scene
- Improved code of the Renderer and reduced code duplication
- Fixed hanging texture references when a texture is deleted

## [0.1.1-beta]

### Added

- Updated external libraries

### Fixed

- Added missing include in system registry

## [0.1.0-beta]

### Added

- Possibility to flip decoded jpeg images
- Custom key codes
- Methods to read and write structs from network buffers
- Add python bindings for network utils
- atcg::shader_directory() and atcg::resource_directory() to get absolute paths to shader and resources
- ATCG_TARGET_DIR for each target that is built to get information about it's location
- A path for atcg::ShaderManagerSystem can now be specified
- Default imgui layout
- Add Worker and WorkerPool for parallalized CPU computing
- The framework can be installed via pip
- Possiblity to index individual layers of a TextureArray
- create method for Cubemaps from tensor data
- Added support for unsigend data types in torch tensors
- Applications can now be set to fullscreen via F11 or via window->toggleFullscreen()
- Add Rendergraph and Renderpasses to implement more complicated rendering algorithms
- Add unit tests for shader and shader manager
- Add function to set window position
- Add functions to change the decoration states of windows
- Add atcg::getCommandLineArguments() to access command line arguments

### Changed

- Default behavior of JPEG encoder and decoder: now flip images
- The caller of TCPClient::sendAndWait had to manually prepend the message size. This is now done automatically.
- The callback functions of TCPServer now expect std::vector references instead of raw pointers.
- Indices are now consistently handles as uint32_t, also for tensors.
- Each Renderer now has it's own instance of a shader manager.
- Each Renderer now has control over it's own context.
- The documentation now uses a nicer theme and is separated into multiple sections.

### Fixed

- Fixed linking of imgui in headless mode by removing unnecessary libraries in linking stage
- Fixed crash when using the JPEG modules without initializing the torch allocator
- Channel size of RG texture format
- isHDR() function for RG textures
- GPU memcopy of float 3D textures
- GPU memcopy of Texture arrays
- Mark non-compute shader as such if a compute shader gets recompiled
