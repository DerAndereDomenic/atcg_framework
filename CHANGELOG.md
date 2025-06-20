# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Note that this project is currently in a beta state. Therefore, there might be API changes that are not reflected in the versioning.

## [Unreleased]

### Added

- Added method to update the near and far plane of the VR System
- Added serialization for InstanceRenderComponents
- Added possibility to attach multiple vbos to a InstanceRenderComponent
- Added python bindings for vertex buffers and related structures
- Added a TextureBuilder class to improve texture instantiation

### Changed

- Removed atcg::Instance
- InstanceRenderComponents now longer are hard typed to atcg::Instance but can be filled with an arbitrary number of instance buffers

### Fixed

- Fixed crash when a Script is held inside the revision system and the app is closed
- Fixed rendering of InstanceRenderComponent

## [0.2.1-beta]

### Fixed

- Fixed bad any_cast in screenshot function

## [0.2.0-beta]

### Added

- Added Cubemap Array textures
- Added possibility to create textures from device tensors
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
- Added context manager for graphics contexts that enables more robust multi-threaded rendering
- Added dedicated shader compiler class
- atcg::CameraController::onUpdate now returns if the camera was updated in the current frame
- Added revision counter to Transform components
- Added copy methods to Graph and Camera
- Added RevisionStack which allows for Ctrl+Z and Ctrl+Y
- Added simple include functionality for shaders
- Added shader subroutines
- Added ATCG_DEBUG for debug messages
- Added explicit models for Camera extrinsics and intrinsics
- Added functions to convert between opengl and opencv intrinsics
- Added width and height to camera component
- Added camera preview in frustum rendering
- Added possibility to load camera images to be rendered
- Added main scene camera
- Added fly-to and set from view camera buttons
- Added scripting engine and python scripting to entities with a Script Component
- Added slider to control the size of rendered camera frustums
- Added PixelUnpackBuffers
- Added possibility to set texture data from PixelUnpackBuffers
- Added options to configure the python side build
- Added draw method to scene
- Added skyboxes to scene
- Added Dictionaries
- Added Skybox class
- Added a more flexible render pass system that allows also the use of predefined renderpass
- Added a component renderer class that handles rendering of different components

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
- Graphics Contexts can no longer be created by hand but have to be created via the ContextManager namespace.
- atcg::CameraController::onUpdate now returns if the camera was updated in the current frame
- drawGuizmo now takes in the scene as an additional parameter
- operator bool() of an entity now also checks if the entity handle is still part of the scene
- IDComponent::id is no longer read-write but read only
- NameComponent::name is no longer read-write but read only
- glm objects can no longer be directly casted to numpy arrays but have to be converted explicitely using .numpy()
- Removed unnecessary wrapper class around logger, now everything directly works on an spdlog object
- Constructor of PerspectiveCamera now uses CameraIntrinsics and CameraExtrinsics
- Camera Controllers can no longer be initiated with an implict camera, but it has to be created and passed manually
- Torch DLLs are NO LONGER COPIED to the output directory on Windows. The user has to make sure the `PATH` variables point to the correct DLLs.
- The `PythonFrontend.py` no longer compiles the project. It now requires that the package is installed via pip.
- The python folder was moved into atcg_lib
- Reworked design of RenderGraph
- The atcg::RendererSystem is now no longer responsible to render scenes and handle cubemaps. This is now done in the scene class
- Renderer::getFramebuffer now returns the internally used framebuffer which might be a MSAA framebuffer if MSAA is enabled. Specific getters for both framebuffers were added

### Fixed

- Screenshot functionality of Renderer uses correct member method and not global renderer instance.
- scene->removeAllEntities now clear internal buffers
- Fixed Texture tests on CPU backend
- Fix normals of cylinder in test scene
- Normals of cylinder in test scene
- Improved code of the Renderer and reduced code duplication
- Fixed hanging texture references when a texture is deleted
- Fixed managing of framebuffer state in multi threaded applications
- Fixed crash when setting faces of a graph with a device tensor (because torch can't sort GPU uint32). Current workaround copies the data to CPU first
- Fixed bindings for glm objects in python
- Fixed GUI for camera extrinsics in SceneHierarchyPanel

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
