import charonload
import pathlib

VSCODE_STUBS_DIRECTORY = pathlib.Path(__file__).parent / "build_python/typings"

charonload.module_config["pyatcg"] = charonload.Config(
    # All paths must be absolute
    project_directory=pathlib.Path(__file__).parent,
    build_directory=pathlib.Path(__file__).parent / "build_python",
    cmake_options={"ATCG_CUDA_BACKEND": "On", "ATCG_PYTHON_BINDINGS" : "On"},
    stubs_directory=VSCODE_STUBS_DIRECTORY,
    build_type="Debug",
    verbose=True
)

import pyatcg as atcg
import torch
import numpy as np

class PythonLayer(atcg.Layer):

    def __init__(self):
        atcg.Layer.__init__(self)

    def onAttach(self):
        atcg.enableDockSpace(True)
        atcg.Renderer.setClearColor(atcg.vec4(0,0,0,1))
        aspect_ratio = atcg.width() / atcg.height()

        self.camera_controller = atcg.FirstPersonController(aspect_ratio)

        self.scene = atcg.Scene()

        self.panel = atcg.SceneHierarchyPanel(self.scene)

        entity = self.scene.createEntity("Sphere")
        self.graph = atcg.read_mesh("res/sphere_low.obj")
        entity.addGeometryComponent(self.graph)
        entity.addTransformComponent(atcg.vec3(0), atcg.vec3(1), atcg.vec3(0))
        entity.addMeshRenderComponent(atcg.ShaderManager.getShader("base"))

        self.current_operation = atcg.ImGui.GuizmoOperation.TRANSLATE

    def onUpdate(self, dt):
        self.camera_controller.onUpdate(dt)

        atcg.Renderer.clear()

        atcg.Renderer.draw(self.scene, self.camera_controller.getCamera())

        atcg.Renderer.drawCameras(self.scene, self.camera_controller.getCamera())

        atcg.Renderer.drawCADGrid(self.camera_controller.getCamera())

    def onImGuiRender(self):
        self.panel.renderPanel()

        selected_entity = self.panel.getSelectedEntity()

        atcg.ImGui.drawGuizmo(selected_entity, self.current_operation, self.camera_controller.getCamera())

    def onEvent(self, event):
        self.camera_controller.onEvent(event)
        
        if event.getName() == "ViewportResize":
            resize_event = atcg.WindowResizeEvent(event.getWidth(), event.getHeight())
            self.camera_controller.onEvent(resize_event)
            

def main():
    layer = PythonLayer()

    props = atcg.WindowProps()
    props.width = 2560 
    props.height = 1440

    atcg.show(layer, props)

main()
atcg.print_statistics()
