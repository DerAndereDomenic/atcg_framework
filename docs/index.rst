.. ATCG Lib documentation master file, created by
   sphinx-quickstart on Tue Aug 22 13:51:09 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ATCG Lib's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`

.. include:: ../README.md
   :parser: myst

Core
====

.. doxygenclass:: atcg::Application
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Input
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Layer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::LayerStack
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::host_allocator
   :members:
   :undoc-members:
.. doxygenclass:: atcg::MemoryContainer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::DevicePointer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::MemoryBuffer
   :members:
   :undoc-members:
.. doxygentypedef:: atcg::scope_ptr
   :project: ATCGLIB
.. doxygentypedef:: atcg::ref_ptr
   :project: ATCGLIB
.. doxygentypedef:: atcg::dref_ptr
   :project: ATCGLIB
.. doxygenfunction:: atcg::make_ref
   :project: ATCGLIB
.. doxygenclass:: atcg::UUID
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::WindowProps
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Window
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::resource_directory
   :project: ATCGLIB
.. doxygenfunction:: atcg::shader_directory
   :project: ATCGLIB

Events
======
.. doxygenenum:: atcg::EventType
   :project: ATCGLIB
.. doxygenenum:: atcg::EventCategory
   :project: ATCGLIB
.. doxygenclass:: atcg::EventDispatcher
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Event
   :members:
   :undoc-members:
.. doxygenclass:: atcg::KeyEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::KeyPressedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::KeyReleasedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::KeyTypedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::MouseMovedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::MouseScrolledEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::MouseButtonEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::MouseButtonPressedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::MouseButtonReleasedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::WindowResizeEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::ViewportResizeEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::WindowCloseEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::FileDroppedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRButtonEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRButtonPressedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRButtonReleasedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRButtonTouchedEvent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRButtonUntouchedEvent
   :members:
   :undoc-members:

DataStructure
=============

.. doxygenclass:: atcg::BufferView
   :members:
   :undoc-members:
.. doxygenenum:: atcg::GraphType
   :project: ATCGLIB
.. doxygenstruct:: atcg::Vertex
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::Edge
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::VertexSpecification
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::EdgeSpecification
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::Instance
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Graph
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::IO::read_any
   :project: ATCGLIB
.. doxygenfunction:: atcg::IO::read_mesh
   :project: ATCGLIB
.. doxygenfunction:: atcg::IO::read_pointcloud
   :project: ATCGLIB
.. doxygenfunction:: atcg::IO::read_lines
   :project: ATCGLIB
.. doxygenfunction:: atcg::IO::read_scene
   :project: ATCGLIB
.. doxygenclass:: atcg::Grid
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Image
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Statistic
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Collection
   :members:
   :undoc-members:
.. doxygenclass:: atcg::CyclicCollection
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Timer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Worker
   :members:
   :undoc-members:
.. doxygenclass:: atcg::WorkerPool
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::TensorOptions::uint8HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int8HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::uint16HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int16HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::uint32HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int32HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::uint64HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int64HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::floatHostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::doubleHostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int8DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::uint8DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::uint16DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int16DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::uint32DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int32DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::uint64DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::int64DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::floatDeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::doubleDeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::HostOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::TensorOptions::DeviceOptions
   :project: ATCGLIB
.. doxygenfunction:: atcg::createTensorFromPointer(void* pointer, const at::IntArrayRef& size, const torch::TensorOptions& options)
   :project: ATCGLIB
.. doxygenfunction:: atcg::createTensorFromPointer(void* pointer, const at::IntArrayRef& size, const at::IntArrayRef& stride, const torch::TensorOptions& options)
   :project: ATCGLIB
.. doxygenfunction:: atcg::createHostTensorFromPointer(T* pointer, const at::IntArrayRef& size)
   :project: ATCGLIB
.. doxygenfunction:: atcg::createHostTensorFromPointer(T* pointer, const at::IntArrayRef& size, const at::IntArrayRef& stride)
   :project: ATCGLIB
.. doxygenfunction:: atcg::createDeviceTensorFromPointer(T* pointer, const at::IntArrayRef& size)
   :project: ATCGLIB
.. doxygenfunction:: atcg::createDeviceTensorFromPointer(T* pointer, const at::IntArrayRef& size, const at::IntArrayRef& stride)
   :project: ATCGLIB
.. doxygenfunction:: atcg::getVertexBufferAsHostTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getVertexBufferAsDeviceTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getPositionsAsHostTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getPositionsAsDeviceTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getColorsAsHostTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getColorsAsDeviceTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getNormalsAsHostTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getNormalsAsDeviceTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getTangentsAsHostTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getTangentsAsDeviceTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getUVsAsHostTensor
   :project: ATCGLIB
.. doxygenfunction:: atcg::getUVsAsDeviceTensor
   :project: ATCGLIB

ImGui
=====
.. doxygenfunction:: atcg::drawGuizmo
   :project: ATCGLIB
.. doxygenclass:: atcg::ImGuiLayer
   :members:
   :undoc-members:

Math
====
.. doxygenfunction:: atcg::Math::map(const T& value, const T& old_left, const T& old_right, const T& new_left, const T& new_right)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::map(const glm::vec<N, T>& value, const glm::vec<N, T>& old_left, const glm::vec<N, T>& old_right, const glm::vec<N, T>& new_left, const glm::vec<N, T>& new_right)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::uv2ndc(const T& val)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::uv2ndc(const glm::vec<N, T>& val)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::ndc2uv(const T& val)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::ndc2uv(const glm::vec<N, T>& val)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::ndc2linearDepth
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::linearDepth2ndc
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::quantize(const glm::vec3& color)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::dequantize(const glm::u8vec3& color)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::quantize(const glm::vec4& color)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::dequantize(const glm::u8vec4& color)
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::sRGB_to_lRGB
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::lRGB_to_sRGB
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::sRGB_to_XYZ
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::XYZ_to_sRGB
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::lRGB_to_XYZ
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::XYZ_to_lRGB
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::lRGB_to_luminance
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::sRGB_to_luminance
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::color_matching_x
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::color_matching_y
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::color_matching_z
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::D65
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::Sr
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::Sg
   :project: ATCGLIB
.. doxygenfunction:: atcg::Color::Sb
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::zero
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::one
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::two_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::root_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::half_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::three_over_two_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::quarter_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::one_over_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::one_over_two_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::two_over_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::four_over_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::two_over_root_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::one_over_root_two
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::root_half_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::root_two_pi
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::root_ln_four
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::e
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::euler
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::root_two
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::root_three
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::root_five
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::ln_two
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::ln_ten
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::ln_ln_two
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::half
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::third
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::two_thirds
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::golden_ratio
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::boltzmann
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::h
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::c
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::G
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::g
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::Y_integral
   :project: ATCGLIB
.. doxygenfunction:: atcg::Constants::Y_illum
   :project: ATCGLIB
.. doxygenfunction:: atcg::normalize(const atcg::ref_ptr<Graph>& graph)
   :project: ATCGLIB
.. doxygenfunction:: atcg::normalize(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform)
   :project: ATCGLIB
.. doxygenfunction:: applyTransform(const atcg::ref_ptr<Graph>& graph, atcg::TransformComponent& transform)
   :project: ATCGLIB
.. doxygenfunction:: applyTransform(torch::Tensor& positions, torch::Tensor& normals, torch::Tensor& tangents, atcg::TransformComponent& transform)
   :project: ATCGLIB
.. doxygenfunction:: atcg::IO::dumpBinary
   :project: ATCGLIB
.. doxygenstruct:: atcg::Tracing::HitInfo
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::Tracing::prepareAccelerationStructure
   :project: ATCGLIB
.. doxygenfunction:: atcg::Tracing::traceRay
   :project: ATCGLIB
.. doxygenfunction:: atcg::ntoh
   :project: ATCGLIB
.. doxygenfunction:: atcg::hton
   :project: ATCGLIB

Network
=======
.. doxygenclass:: atcg::TCPServer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::TCPClient
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::NetworkUtils::readByte
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::readInt
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::readString
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::readStruct
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::writeByte
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::writeInt
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::writeBuffer
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::writeString
   :project: ATCGLIB
.. doxygenfunction:: atcg::NetworkUtils::writeStruct
   :project: ATCGLIB

OpenMesh
========
.. doxygentypedef:: atcg::TriMesh
   :project: ATCGLIB
.. doxygentypedef:: atcg::PolyMesh
   :project: ATCGLIB

Renderer
========
.. doxygenclass:: atcg::Context
   :members:
   :undoc-members:
.. doxygenenum:: atcg::ShaderDataType
   :project: ATCGLIB
.. doxygenfunction:: atcg::ShaderDataTypeSize
   :project: ATCGLIB
.. doxygenstruct:: atcg::BufferElement
   :members:
   :undoc-members:
.. doxygenclass:: atcg::BufferLayout
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VertexBuffer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::IndexBuffer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VertexArray
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Camera
   :members:
   :undoc-members:
.. doxygenclass:: atcg::OrthographicCamera
   :members:
   :undoc-members:
.. doxygenclass:: atcg::PerspectiveCamera
   :members:
   :undoc-members:
.. doxygenclass:: atcg::CameraController
   :members:
   :undoc-members:
.. doxygenclass:: atcg::FocusedController
   :members:
   :undoc-members:
.. doxygenclass:: atcg::FirstPersonController
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRController
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Framebuffer
   :members:
   :undoc-members:
.. doxygenenum:: atcg::TextureFormat
   :project: ATCGLIB
.. doxygenenum:: atcg::TextureWrapMode
   :project: ATCGLIB
.. doxygenenum:: atcg::TextureFilterMode
   :project: ATCGLIB
.. doxygenstruct:: atcg::TextureSampler
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::TextureSpecification
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Texture
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Texture2D
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Texture3D
   :members:
   :undoc-members:
.. doxygenclass:: atcg::TextureCube
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Shader
   :members:
   :undoc-members:
.. doxygenclass:: atcg::ShaderManagerSystem
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::Material
   :members:
   :undoc-members:
.. doxygenenum:: atcg::DrawMode
   :project: ATCGLIB
.. doxygenenum:: atcg::CullMode
   :project: ATCGLIB
.. doxygenclass:: atcg::RendererSystem
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRSystem
   :members:
   :undoc-members:

Scene
=====
.. doxygenstruct:: atcg::TransformComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::IDComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::NameComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::GeometryComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::AccelerationStructureComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::CameraComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::EditorCameraComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::RenderComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::MeshRenderComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::PointRenderComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::PointSphereRenderComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::EdgeRenderComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::EdgeCylinderRenderComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::InstanceRenderComponent
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::CustomRenderComponent
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Entity
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Scene
   :members:
   :undoc-members:
.. doxygenclass:: atcg::SceneHierarchyPanel
   :members:
   :undoc-members:
.. doxygenclass:: atcg::ComponentGUIHandler
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Serializer
   :members:
   :undoc-members:
.. doxygenclass:: atcg::ComponentSerializer
   :members:
   :undoc-members: