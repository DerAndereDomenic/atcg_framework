# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed

- Default behavior of JPEG encoder and decoder: now flip images
- The caller of TCPClient::sendAndWait had to manually prepend the message size. This is now done automatically.
- The callback functions of TCPServer now expect std::vector references instead of raw pointers.
- Indices are now consistently handles as uint32_t, also for tensors.

### Fixed

- Fixed linking of imgui in headless mode by removing unnecessary libraries in linking stage
- Fixed crash when using the JPEG modules without initializing the torch allocator
- Channel size of RG texture format
- isHDR() function for RG textures
- GPU memcopy of float 3D textures
- GPU memcopy of Texture arrays
