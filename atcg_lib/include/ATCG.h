#pragma once

//-------- CORE ---------
#include <Core/Layer.h>
#include <Core/LayerStack.h>
#include <Core/Application.h>
#include <Core/Window.h>
#include <Core/Input.h>
#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/Log.h>
#include <Core/CUDA.h>

//-------- EVENTS -------
#include <Events/Event.h>
#include <Events/MouseEvent.h>
#include <Events/WindowEvent.h>
#include <Events/KeyEvent.h>

//-------- Renderer ------
#include <Renderer/Renderer.h>
#include <Renderer/Shader.h>
#include <Renderer/ShaderManager.h>
#include <Renderer/PerspectiveCamera.h>
#include <Renderer/OrthographicCamera.h>
#include <Renderer/CameraController.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Texture.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/Radiosity.h>

//-------- OpenMesh -------
// #include <OpenMesh/OpenMesh.h>

//-------- DataStructure --
#include <DataStructure/Mesh.h>
#include <DataStructure/Grid.h>
#include <DataStructure/Laplacian.h>
#include <DataStructure/Timer.h>
#include <DataStructure/PointCloud.h>
#include <DataStructure/Statistics.h>

//-------- Math -----------
#include <Math/Functions.h>
#include <Math/Utils.h>
#include <Math/Tracing.h>

//-------- Utility -----------
#include <Noise/Noise.h>