# RUN: pip install -e . before running this file

import torch
import numpy as np
import matplotlib.pyplot as plt
import platform
from tqdm import tqdm

import os

os.environ["ATCG_BUILD_TYPE"] = "RelWithDebInfo"

if torch.cuda.is_available():
    os.environ["ATCG_CUDA_BACKEND"] = "On"
else:
    os.environ["ATCG_CUDA_BACKEND"] = "Off"

if torch.cuda.is_available() and "Linux" in platform.platform():
    os.environ["ATCG_HEADLESS"] = "On"

import pyatcg as atcg


def render_scene(integrator, rng_index, n_samples):
    prediction = 0
    for _ in range(n_samples):
        prediction = prediction + integrator.generateRays(rng_index)
        rng_index += 1

    prediction /= n_samples
    return prediction, rng_index


def run_method(method_name, context, scene, width, height, parameter_name):

    integrator = atcg.IntegratorRegistry.createIntegrator(
        method_name, context, scene, width, height
    )

    object_name, component, parameter = parameter_name.split(".")

    target, rng_index = render_scene(integrator, 0, 1024)

    optix_scene = integrator.getOptixScene()

    objects = optix_scene.getEntitiesByName(object_name)

    if component == "bsdf":
        bsdf = objects[0].getBSDFComponent()
        material = bsdf.bsdf
    elif component == "medium":
        medium = objects[0].getMediumComponent()
        material = medium.medium

    parameter_tensor = torch.tensor(
        [[0.8]], dtype=torch.float32, device="cuda", requires_grad=True
    )
    material.setParameter(parameter, parameter_tensor)

    with torch.no_grad():
        initial, rng_index = render_scene(integrator, rng_index, 128)

    n_epochs = 256

    loss_values = []
    roughness_values = []
    roughness_grad_values = []

    optimizer = torch.optim.Adam([material.getParameter(parameter)], lr=0.01)

    for _ in tqdm(range(n_epochs), desc="Optimizing"):
        optimizer.zero_grad()
        prediction, rng_index = render_scene(integrator, rng_index, 128)

        with torch.no_grad():
            fake_prediction, rng_index = render_scene(integrator, rng_index, 128)

        prediction_injected = prediction + (fake_prediction - prediction).detach()

        L = torch.sum(torch.abs(prediction_injected - target) ** 2)

        L.backward()
        optimizer.step()

        with torch.no_grad():
            if parameter == "roughness_texture":
                material.getParameter(parameter).clamp_(0.0, 1.0)
            elif parameter == "density":
                material.getParameter(parameter).clamp_(0.0)

        r = material.getParameter(parameter)
        dr = r.grad

        roughness_values.append(r.detach().cpu().numpy().item())
        roughness_grad_values.append(dr.detach().cpu().numpy().item())
        loss_values.append(L.detach().cpu().numpy().item())

    fig, axes = plt.subplots(
        3, 2, figsize=(12, 10), gridspec_kw={"width_ratios": [1, 1.5]}
    )

    # Left column: Images
    axes[0, 0].imshow(target.detach().cpu().flip(0))
    axes[0, 0].set_title("Target")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(initial.detach().cpu().flip(0))
    axes[1, 0].set_title("Initial Prediction")
    axes[1, 0].axis("off")

    # Empty bottom-left
    axes[2, 0].imshow(prediction.detach().cpu().flip(0))
    axes[2, 0].set_title("Prediction")
    axes[2, 0].axis("off")

    # Right column: Graphs
    axes[0, 1].plot(loss_values)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Optimization Progress")

    axes[1, 1].plot(roughness_values)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Roughness Value")
    axes[1, 1].set_title("Roughness Value Over Epochs")

    axes[2, 1].plot(roughness_grad_values)
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Roughness Gradient")
    axes[2, 1].set_title("Roughness Gradient Over Epochs")

    plt.tight_layout()
    plt.show()


def run_test_scene(context, width, height):
    atcg.Project.load("../DiffRendTest_old/Project.json")
    scene = atcg.Project.getActive().getActiveScene()

    extrinsics = atcg.CameraExtrinsics()
    extrinsics.setPosition(atcg.vec3(0.0, 1.0, 3.0))
    extrinsics.setTarget(atcg.vec3(0.0, 1.0, 0.0))
    intrinsics = atcg.CameraIntrinsics()
    intrinsics.setAspectRatio(width / height)
    camera = atcg.PerspectiveCamera(extrinsics, intrinsics)
    scene.setCamera(camera)

    parameter_name = "sphere.bsdf.roughness_texture"

    method_names = [
        "RBPIntegrator",
        "DiffPathtracingIntegrator",
        "AttachedDiffPathtracingIntegrator",
    ]

    for method_name in method_names:
        print(f"Running optimization for method: {method_name}")
        run_method(method_name, context, scene, width, height, parameter_name)


def run_suzanne_scene(context, width, height):
    atcg.Project.load("../CornellBoxOptimization/Project.json")
    scene = atcg.Project.getActive().getActiveScene()

    cam = scene.getEntitiesByName("Empty Entity")[0].getCameraComponent().camera()

    extrinsics = atcg.CameraExtrinsics()
    extrinsics.setPosition(cam.getPosition())
    extrinsics.setTarget(cam.getLookAt())
    intrinsics = atcg.CameraIntrinsics()
    intrinsics.setAspectRatio(width / height)
    camera = atcg.PerspectiveCamera(extrinsics, intrinsics)
    scene.setCamera(camera)

    parameter_name = "suzanne.bsdf.roughness_texture"

    method_names = [
        "RBPIntegrator",
        "DiffPathtracingIntegrator",
        "AttachedDiffPathtracingIntegrator",
    ]

    for method_name in method_names:
        print(f"Running optimization for method: {method_name}")
        run_method(method_name, context, scene, width, height, parameter_name)


def run_hom_volume_scene(context, width, height):
    atcg.Project.load("../MediumTestEdge/Project.json")
    scene = atcg.Project.getActive().getActiveScene()

    extrinsics = atcg.CameraExtrinsics()
    extrinsics.setPosition(atcg.vec3(0.0, 1.0, 3.0))
    extrinsics.setTarget(atcg.vec3(0.0, 1.0, 0.0))
    intrinsics = atcg.CameraIntrinsics()
    intrinsics.setAspectRatio(width / height)
    camera = atcg.PerspectiveCamera(extrinsics, intrinsics)
    scene.setCamera(camera)

    parameter_name = "Medium.medium.density"

    method_names = [
        "VolDiffPathtracingIntegrator",
        "VolAttachedDiffPathtracingIntegrator",
    ]

    for method_name in method_names:
        print(f"Running optimization for method: {method_name}")
        run_method(method_name, context, scene, width, height, parameter_name)


def main():

    width = 128
    height = 128

    props = atcg.WindowProps()
    props.width = width
    props.height = height
    props.hidden = True

    app = atcg.PythonApplication(props)
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/DetachedDiffPath.dll")
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/AttachedDiffPath.dll")
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/RBPDiffPath.dll")
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/VolAttachedDiffPath.dll")
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/VolDetachedDiffPath.dll")

    context = atcg.RaytracingContextManager.createContext(0)

    run_test_scene(context, width, height)
    run_suzanne_scene(context, width, height)
    run_hom_volume_scene(context, width, height)


main()
