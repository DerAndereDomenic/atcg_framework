#pragma once

//-------- CORE ---------
#include <Core/Layer.h>
#include <Core/LayerStack.h>
#include <Core/Application.h>
#include <Core/Window.h>
#include <Core/Input.h>
#include <Core/API.h>

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

//-------- OpenMesh -------
//#include <OpenMesh/OpenMesh.h>

//-------- DataStructure --
#include <DataStructure/Mesh.h>
#include <DataStructure/Grid.h>
#include <DataStructure/Laplacian.h>
#include <DataStructure/Timer.h>