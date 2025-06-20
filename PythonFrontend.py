# RUN: pip install -e . before running this file

import torch
import numpy as np


import pyatcg as atcg


class PythonLayer(atcg.Layer):

    def __init__(self):
        atcg.Layer.__init__(self)

    def create_instance(self):
        mesh = atcg.read_mesh(f"{atcg.resource_directory()}/sphere_low.obj")
        n_instances = 100

        transforms = []
        colors = []

        for _ in range(n_instances):
            pos = np.random.uniform(-5, 5, 3).astype(np.float32)
            pos[1] += 10
            m = np.eye(4).astype(np.float32)
            m[-1, :-1] = pos

            R = np.random.randn(3, 3)

            # Symmetric positive-definite matrix
            cov = R @ R.T

            # Normalize to unit max eigenvalue
            eigvals = np.linalg.eigvalsh(cov)
            max_eig = np.max(eigvals)
            cov_normalized = cov / max_eig

            # Scale to desired max variance
            cov_scaled = cov_normalized * 0.3

            m[:3, :3] = cov_scaled

            transforms.append(m)
            color = np.random.uniform(0, 1, 4).astype(np.float32)
            color[-1] = 1
            colors.append(color)

        transforms = np.array(transforms)
        colors = np.array(colors)

        layout_transforms = atcg.BufferLayout(
            [atcg.BufferElement(atcg.ShaderDataType.Mat4, "Transform")]
        )
        vbo_transforms = atcg.VertexBuffer(transforms)
        vbo_transforms.setLayout(layout_transforms)
        layout_colors = atcg.BufferLayout(
            [atcg.BufferElement(atcg.ShaderDataType.Float4, "Color")]
        )
        vbo_colors = atcg.VertexBuffer(colors)
        vbo_colors.setLayout(layout_colors)

        entity = self.scene.createEntity("Instances")
        entity.addTransformComponent(atcg.vec3(0), atcg.vec3(1), atcg.vec3(0))
        entity.addGeometryComponent(mesh)
        instances = entity.addInstanceRenderComponent()

        instances.instances = [vbo_transforms, vbo_colors]

        entity.replaceInstanceRenderComponent(instances)

    def onAttach(self):
        atcg.enableDockSpace(True)
        atcg.Renderer.setClearColor(atcg.vec4(0, 0, 0, 1))
        aspect_ratio = atcg.width() / atcg.height()

        extrinsics = atcg.CameraExtrinsics()
        intrinsics = atcg.CameraIntrinsics()
        intrinsics.setAspectRatio(aspect_ratio)
        camera = atcg.PerspectiveCamera(extrinsics, intrinsics)
        self.camera_controller = atcg.FirstPersonController(camera)

        self.scene = atcg.Scene()

        self.panel = atcg.SceneHierarchyPanel(self.scene)
        self.performance_panel = atcg.PerformancePanel()
        entity = self.scene.createEntity("Cylinder")
        self.graph = atcg.read_mesh(f"{atcg.resource_directory()}/cylinder.obj")
        entity.addGeometryComponent(self.graph)
        entity.addTransformComponent(atcg.vec3(0), atcg.vec3(1), atcg.vec3(0))
        renderer = entity.addMeshRenderComponent(atcg.ShaderManager.getShader("base"))

        diffuse_img = atcg.imread(f"{atcg.resource_directory()}/pbr/diffuse.png", 2.2)
        normal_img = atcg.imread(f"{atcg.resource_directory()}/pbr/normals.png", 1.0)
        roughness_img = atcg.imread(
            f"{atcg.resource_directory()}/pbr/roughness.png", 1.0
        )
        metallic_img = atcg.imread(f"{atcg.resource_directory()}/pbr/metallic.png", 1.0)

        diffuse_texture = atcg.Texture2D.create(diffuse_img)
        normal_texture = atcg.Texture2D.create(normal_img)
        roughness_texture = atcg.Texture2D.create(roughness_img)
        metallic_texture = atcg.Texture2D.create(metallic_img)

        renderer.material.setDiffuseTexture(diffuse_texture)
        renderer.material.setNormalTexture(normal_texture)
        renderer.material.setRoughnessTexture(roughness_texture)
        renderer.material.setMetallicTexture(metallic_texture)
        entity.replaceMeshRenderComponent(renderer)

        self.create_instance()

        self.current_operation = atcg.ImGui.GuizmoOperation.TRANSLATE

    def onUpdate(self, dt):
        self.performance_panel.registerFrameTime(dt)
        self.camera_controller.onUpdate(dt)

        atcg.handleScriptUpdates(self.scene, dt)

        atcg.Renderer.clear()

        self.scene.draw(self.camera_controller.getCamera())

        atcg.Renderer.drawCameras(self.scene, self.camera_controller.getCamera())

        atcg.Renderer.drawCADGrid(self.camera_controller.getCamera())

    def onImGuiRender(self):
        self.panel.renderPanel()

        selected_entity = self.panel.getSelectedEntity()

        self.performance_panel.renderPanel(True)

        atcg.ImGui.drawGuizmo(
            self.scene,
            selected_entity,
            self.current_operation,
            self.camera_controller.getCamera(),
        )

    def onEvent(self, event):
        self.camera_controller.onEvent(event)

        atcg.handleScriptEvents(self.scene, event)

        if event.getName() == "ViewportResize":
            resize_event = atcg.WindowResizeEvent(event.getWidth(), event.getHeight())
            self.camera_controller.onEvent(resize_event)


def main():
    props = atcg.WindowProps()
    props.width = 1600
    props.height = 900

    layer = PythonLayer()
    app = atcg.PythonApplication(layer, props)

    app.run()


main()
