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
.. doxygenclass:: atcg::Input
   :members:
.. doxygenclass:: atcg::Layer
   :members:
.. doxygenclass:: atcg::LayerStack
   :members:
.. doxygenstruct:: atcg::host_allocator
   :members:
.. doxygenclass:: atcg::MemoryContainer
   :members:
.. doxygenclass:: atcg::DeviceBuffer
   :members:
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
.. doxygenstruct:: atcg::WindowProps
   :members:
.. doxygenclass:: atcg::Window
   :members:

Events
======
.. doxygenenum:: atcg::EventType
   :project: ATCGLIB
.. doxygenenum:: atcg::EventCategory
   :project: ATCGLIB
.. doxygenclass:: atcg::EventDispatcher
   :members:
.. doxygenclass:: atcg::Event
   :members:
.. doxygenclass:: atcg::KeyEvent
   :members:
.. doxygenclass:: atcg::KeyPressedEvent
   :members:
.. doxygenclass:: atcg::KeyReleasedEvent
   :members:
.. doxygenclass:: atcg::KeyTypedEvent
   :members:
.. doxygenclass:: atcg::MouseMovedEvent
   :members:
.. doxygenclass:: atcg::MouseScrolledEvent
   :members:
.. doxygenclass:: atcg::MouseButtonEvent
   :members:
.. doxygenclass:: atcg::MouseButtonPressedEvent
   :members:
.. doxygenclass:: atcg::MouseButtonReleasedEvent
   :members:
.. doxygenclass:: atcg::WindowResizeEvent
   :members:
.. doxygenclass:: atcg::ViewportResizeEvent
   :members:
.. doxygenclass:: atcg::WindowCloseEvent
   :members:
.. doxygenclass:: atcg::FileDroppedEvent
   :members:

DataStructure
=============

.. .. doxygenclass:: atcg::BufferView
   :members:
.. doxygenenum:: atcg::GraphType
   :project: ATCGLIB
.. doxygenstruct:: atcg::Vertex
   :members:
.. doxygenstruct:: atcg::Edge
   :members:
.. doxygenstruct:: atcg::Instance
   :members:
.. doxygenclass:: atcg::Graph
   :members:
.. doxygenfunction:: atcg::IO::read_mesh
   :project: ATCGLIB
.. doxygenclass:: atcg::Grid
   :members:
.. doxygenclass:: atcg::Statistic
   :members:
.. doxygenclass:: atcg::Timer
   :members:

ImGui
=====
.. doxygenfunction:: atcg::drawGuizmo
   :project: ATCGLIB
.. doxygenclass:: atcg::ImGuiLayer
   :members:

Math
====
.. doxygenfunction:: atcg::normalize
   :project: ATCGLIB
.. doxygenstruct:: atcg::Tracing::HitInfo
   :members:
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
.. doxygenclass:: atcg::BufferLayout
   :members:
.. doxygenclass:: atcg::VertexBuffer
   :members:
.. doxygenclass:: atcg::IndexBuffer
   :members:
.. doxygenclass:: atcg::VertexArray
   :members:
.. doxygenclass:: atcg::Camera
   :members:
.. doxygenclass:: atcg::OrthographicCamera
   :members:
.. doxygenclass:: atcg::PerspectiveCamera
   :members:
.. doxygenclass:: atcg::CameraController
   :members:
.. doxygenclass:: atcg::FocusedController
   :members:
.. doxygenclass:: atcg::FirstPersonController
   :members:
.. doxygenclass:: atcg::Framebuffer
   :members:
.. doxygenclass:: atcg::Texture
   :members:
.. doxygenclass:: atcg::Texture2D
   :members:
.. doxygenclass:: atcg::Texture3D
   :members:
.. doxygenclass:: atcg::Shader
   :members:
.. doxygenclass:: atcg::ShaderManager
   :members:
.. doxygenenum:: atcg::DrawMode
   :project: ATCGLIB
.. doxygenenum:: atcg::CullMode
   :project: ATCGLIB
.. doxygenclass:: atcg::Renderer
   :members:

Scene
=====
.. doxygenstruct:: atcg::TransformComponent
   :members:
.. doxygenstruct:: atcg::IDComponent
   :members:
.. doxygenstruct:: atcg::NameComponent
   :members:
.. doxygenstruct:: atcg::GeometryComponent
   :members:
.. doxygenstruct:: atcg::AccelerationStructureComponent
   :members:
.. doxygenstruct:: atcg::CameraComponent
   :members:
.. doxygenstruct:: atcg::EditorCameraComponent
   :members:
.. doxygenstruct:: atcg::RenderComponent
   :members:
.. doxygenstruct:: atcg::TextureComponent
   :members:
.. doxygenstruct:: atcg::MeshRenderComponent
   :members:
.. doxygenstruct:: atcg::PointRenderComponent
   :members:
.. doxygenstruct:: atcg::PointSphereRenderComponent
   :members:
.. doxygenstruct:: atcg::EdgeRenderComponent
   :members:
.. doxygenstruct:: atcg::EdgeCylinderRenderComponent
   :members:
.. doxygenstruct:: atcg::InstanceRenderComponent
   :members:
.. doxygenstruct:: atcg::CustomRenderComponent
   :members:
.. doxygenclass:: atcg::Entity
   :members:
.. doxygenclass:: atcg::Scene
   :members:
.. doxygenclass:: atcg::SceneHierarchyPanel
   :members:
.. doxygenclass:: atcg::Serializer
   :members: