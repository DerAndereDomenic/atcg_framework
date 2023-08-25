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
* :ref:`search`


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
.. doxygenclass:: atcg::DeviceBuffer
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

DataStructure
=============

.. .. doxygenclass:: atcg::BufferView
   :members:
.. doxygenenum:: atcg::GraphType
   :project: ATCGLIB
.. doxygenstruct:: atcg::Vertex
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::Edge
   :members:
   :undoc-members:
.. doxygenstruct:: atcg::Instance
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Graph
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::IO::read_mesh
   :project: ATCGLIB
.. doxygenclass:: atcg::Grid
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Statistic
   :members:
   :undoc-members:
.. doxygenclass:: atcg::Timer
   :members:
   :undoc-members:

ImGui
=====
.. doxygenfunction:: atcg::drawGuizmo
   :project: ATCGLIB
.. doxygenclass:: atcg::ImGuiLayer
   :members:
   :undoc-members:

Math
====
.. doxygenfunction:: atcg::normalize
   :project: ATCGLIB
.. doxygenstruct:: atcg::Tracing::HitInfo
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::Tracing::prepareAccelerationStructure
   :project: ATCGLIB
.. doxygenfunction:: atcg::Tracing::traceRay
   :project: ATCGLIB
.. doxygenfunction:: atcg::areaFromMetric
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::sphericalHarmonic
   :project: ATCGLIB
.. doxygenfunction:: atcg::Math::ceil_div
   :project: ATCGLIB

Noise
=====
.. doxygenfunction:: atcg::Noise::createWhiteNoiseTexture2D
   :project: ATCGLIB
.. doxygenfunction:: atcg::Noise::createWhiteNoiseTexture3D
   :project: ATCGLIB
.. doxygenfunction:: atcg::Noise::createWorleyNoiseTexture2D
   :project: ATCGLIB
.. doxygenfunction:: atcg::Noise::createWorleyNoiseTexture3D
   :project: ATCGLIB

OpenMesh
========
.. doxygentypedef:: atcg::TriMesh
   :project: ATCGLIB
.. doxygentypedef:: atcg::PolyMesh
   :project: ATCGLIB

Renderer
========
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
.. doxygenclass:: atcg::Framebuffer
   :members:
   :undoc-members:
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
.. doxygenclass:: atcg::Shader
   :members:
   :undoc-members:
.. doxygenclass:: atcg::ShaderManager
   :members:
   :undoc-members:
.. doxygenenum:: atcg::DrawMode
   :project: ATCGLIB
.. doxygenenum:: atcg::CullMode
   :project: ATCGLIB
.. doxygenclass:: atcg::Renderer
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
.. doxygenstruct:: atcg::TextureComponent
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
.. doxygenclass:: atcg::Serializer
   :members:
   :undoc-members: