#pragma once

//-------- CORE ---------
#include <Core/Assert.h>
#include <Core/Platform.h>
#include <Core/Layer.h>
#include <Core/LayerStack.h>
#include <Core/Application.h>
#include <Core/Window.h>
#include <Core/Input.h>
#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/Log.h>
#include <Core/CUDA.h>
#include <Core/glm.h>
#include <Core/UUID.h>
#include <Core/SystemRegistry.h>
#include <Core/KeyCodes.h>
#include <Core/Path.h>
#include <Core/CommandLine.h>

//-------- EVENTS -------
#include <Events/Event.h>
#include <Events/MouseEvent.h>
#include <Events/WindowEvent.h>
#include <Events/KeyEvent.h>
#include <Events/VREvent.h>

//-------- Renderer ------
#include <Renderer/Context.h>
#include <Renderer/ContextManager.h>
#include <Renderer/Renderer.h>
#include <Renderer/Shader.h>
#include <Renderer/ShaderCompiler.h>
#include <Renderer/ShaderManager.h>
#include <Renderer/PerspectiveCamera.h>
#include <Renderer/OrthographicCamera.h>
#include <Renderer/CameraController.h>
#include <Renderer/Buffer.h>
#include <Renderer/VertexArray.h>
#include <Renderer/Texture.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/Material.h>
#include <Renderer/VRSystem.h>
#include <Renderer/RenderGraph.h>

//-------- OpenMesh -------
// #include <OpenMesh/OpenMesh.h>

//-------- DataStructure --
#include <DataStructure/Graph.h>
#include <DataStructure/GraphLoader.h>
#include <DataStructure/Grid.h>
#include <DataStructure/Timer.h>
#include <DataStructure/Statistics.h>
#include <DataStructure/BufferView.h>
#include <DataStructure/Image.h>
#include <DataStructure/Worker.h>
#include <DataStructure/WorkerPool.h>
#include <DataStructure/PerformancePanel.h>
#ifdef ATCG_CUDA_BACKEND
    #include <DataStructure/JPEGDecoder.h>
    #include <DataStructure/JPEGEncoder.h>
#endif

//-------- Math -----------
#include <Math/Constants.h>
#include <Math/Functions.h>
#include <Math/Utils.h>
#include <Math/Tracing.h>
#include <Math/Color.h>

//-------- Network ------------
#include <Network/TCPServer.h>
#include <Network/TCPClient.h>
#include <Network/NetworkUtils.h>

//-------- ImGui -----------
#ifndef ATCG_HEADLESS
    #include <imgui.h>
    #include <ImGui/Guizmo.h>
#endif

//-------- Scene -----------
#include <Scene/Components.h>
#include <Scene/Scene.h>
#include <Scene/Entity.h>
#include <Scene/Serializer.h>
#include <Scene/SceneHierarchyPanel.h>
#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>
#include <Scene/RevisionStack.h>

//-------- Torch ------------
#include <DataStructure/TorchUtils.h>

//-------- Optix ------------
#ifdef ATCG_ENABLE_OPTIX
    #include <Core/RaytracingPipeline.h>
    #include <Core/ShaderBindingTable.h>
    #include <Integrator/PathtracingIntegrator.h>
#endif

//-------- Scripting ------------
#include <Scripting/ScriptEngine.h>
#include <Scripting/Script.h>
