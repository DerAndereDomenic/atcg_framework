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

n = 16


def render_scene(integrator, rng_index, n_samples):
    prediction = 0
    for _ in tqdm(range(n_samples)):
        prediction = prediction + integrator.generateRays(rng_index)
        rng_index += 1

    prediction /= n_samples
    return prediction, rng_index


def tonemap(hdr_image):
    ldr = 1.0 - torch.exp(-hdr_image)
    ldr = torch.pow(ldr, 1.0 / 2.4)
    ldr = torch.clamp(ldr, 0.0, 1.0)
    return (ldr * 255.0).to(torch.uint8)


def estimate_gradient_fd(integrator: atcg.Integrator):
    print("Finite difference gradient estimation")
    optix_scene = integrator.getOptixScene()

    objects = optix_scene.getEntitiesByName("sphere")

    medium = objects[0].getMediumComponent().medium

    roughness = medium.getParameter("density")

    h = 0.001
    roughness_new = roughness + h
    medium.setParameter("density", roughness_new)
    target_p, _ = render_scene(integrator, 0, 1024 * n)
    roughness_new = roughness - h
    medium.setParameter("density", roughness_new)
    target_m, _ = render_scene(integrator, 0, 1024 * n)
    # plt.imshow(target_m.detach().cpu().flip(0).numpy())
    # plt.show()
    target_p = torch.sum(target_p, dim=-1)
    target_m = torch.sum(target_m, dim=-1)

    dImage_droughness = (target_p - target_m) / (2 * h)

    # plt.imshow(target_p.detach().cpu().flip(0).numpy())
    # plt.show()

    # Restore
    medium.setParameter("density", roughness)

    return dImage_droughness


def estimate_gradient_prbp(integrator: atcg.Integrator):
    print("Detached Path Replay Backpropagation")
    optix_scene = integrator.getOptixScene()

    objects = optix_scene.getEntitiesByName("sphere")

    medium = objects[0].getMediumComponent().medium

    roughness = medium.getParameter("density")

    roughness_new = roughness.clone().requires_grad_(True)
    medium.setParameter("density", roughness_new)

    num_samples = 1024 * n
    rng_index = 0
    dImage_droughness = 0
    L = 0
    for _ in tqdm(range(num_samples)):
        dictionary = atcg.Dictionary()
        dictionary.setUInt32("rng_index", rng_index)
        current_sample = integrator.forwardTrace(dictionary)
        rng_index += 1

        # L += current_sample

        dictionary.setTensor("adjoint_y", torch.ones_like(current_sample))
        dictionary.setTensor("current_sample", current_sample)

        integrator.backwardTrace(dictionary)

        dImage_droughness += integrator.getAOVBuffer(0)

    # plt.imshow(L.detach().cpu().flip(0).numpy() / num_samples)
    # plt.show()

    return dImage_droughness / num_samples


def estimate_gradient_atprbp(integrator: atcg.Integrator):
    print("Attached Path Replay Backpropagation")
    optix_scene = integrator.getOptixScene()

    objects = optix_scene.getEntitiesByName("sphere")

    medium = objects[0].getMediumComponent().medium

    roughness = medium.getParameter("density")

    roughness_new = roughness.clone().requires_grad_(True)
    medium.setParameter("density", roughness_new)

    num_samples = 1024 * n
    rng_index = 0
    dImage_droughness = 0
    L = 0
    for _ in tqdm(range(num_samples)):
        dictionary = atcg.Dictionary()
        dictionary.setUInt32("rng_index", rng_index)
        current_sample, current_JL = integrator.forwardTrace(dictionary)
        rng_index += 1

        dictionary.setTensor("adjoint_y", torch.ones_like(current_sample))
        dictionary.setTensor("current_sample", current_sample)
        dictionary.setTensor("JL_buffer", current_JL)

        L += current_sample

        integrator.backwardTrace(dictionary)

        dImage_droughness += integrator.getAOVBuffer(0)

    tonemapped = tonemap(L / num_samples)

    plt.imshow(tonemapped.detach().cpu().flip(0).numpy())
    plt.show()

    return dImage_droughness / num_samples


# def estimate_gradient_rbp(integrator: atcg.Integrator):
#     print("Radiative Backpropagation")
#     optix_scene = integrator.getOptixScene()

#     objects = optix_scene.getEntitiesByName("sphere")

#     bsdf = objects[0].getBSDFComponent().bsdf

#     roughness = bsdf.getParameter("density")

#     roughness_new = roughness.clone().requires_grad_(True)
#     bsdf.setParameter("density", roughness_new)

#     num_samples = 1024 * n
#     rng_index = 0
#     dImage_droughness = 0
#     for _ in tqdm(range(num_samples)):
#         dictionary = atcg.Dictionary()
#         dictionary.setUInt32("rng_index", rng_index)
#         current_sample = integrator.forwardTrace(dictionary)
#         rng_index += 1

#         dictionary.setTensor("adjoint_y", torch.ones_like(current_sample))
#         dictionary.setTensor("current_sample", current_sample)

#         integrator.backwardTrace(dictionary)

#         dImage_droughness += integrator.getAOVBuffer(0)

#     return dImage_droughness / num_samples


def main():

    width = 128
    height = 128

    props = atcg.WindowProps()
    props.width = width
    props.height = height
    props.hidden = True

    app = atcg.PythonApplication(props)
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/VolDetachedDiffPath.dll")
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/VolAttachedDiffPath.dll")

    context = atcg.RaytracingContextManager.createContext(0)

    atcg.Project.load("../MediumEdge/Project.json")
    scene = atcg.Project.getActive().getActiveScene()

    extrinsics = atcg.CameraExtrinsics()
    extrinsics.setPosition(atcg.vec3(0.0, 1.0, 3.0))
    extrinsics.setTarget(atcg.vec3(0.0, 1.0, 0.0))
    intrinsics = atcg.CameraIntrinsics()
    intrinsics.setAspectRatio(width / height)
    camera = atcg.PerspectiveCamera(extrinsics, intrinsics)
    scene.setCamera(camera)

    # atcg.Project.load("../CornellBoxOptimization/Project.json")
    # scene = atcg.Project.getActive().getActiveScene()

    # cam = scene.getEntitiesByName("Empty Entity")[0].getCameraComponent().camera()

    # extrinsics = atcg.CameraExtrinsics()
    # extrinsics.setPosition(cam.getPosition())
    # extrinsics.setTarget(cam.getLookAt())
    # intrinsics = atcg.CameraIntrinsics()
    # intrinsics.setAspectRatio(width / height)
    # camera = atcg.PerspectiveCamera(extrinsics, intrinsics)
    # scene.setCamera(camera)

    dictionary = atcg.Dictionary()
    dictionary.setUInt32("num_aovs", 1)
    integrator_det = atcg.IntegratorRegistry.createIntegrator(
        "VolDiffPathtracingIntegrator", context, scene, width, height, dictionary
    )

    integrator_att = atcg.IntegratorRegistry.createIntegrator(
        "VolAttachedDiffPathtracingIntegrator",
        context,
        scene,
        width,
        height,
        dictionary,
    )

    # integrator_rbp = atcg.IntegratorRegistry.createIntegrator(
    #     "RBPIntegrator", context, scene, width, height, dictionary
    # )

    # dImage_droughness_fd = estimate_gradient_fd(integrator_det)
    # dImage_droughness_rbp = estimate_gradient_rbp(integrator_rbp)
    # dImage_droughness_prbp = estimate_gradient_prbp(integrator_det)
    dImage_droughness_atprbp = estimate_gradient_atprbp(integrator_att)

    # v_fd = torch.max(torch.abs(dImage_droughness_fd)).item()
    # v_prbp = torch.max(torch.abs(dImage_droughness_prbp)).item()
    # v_atprbp = torch.max(torch.abs(dImage_droughness_atprbp)).item()

    v = 0.2  # max(v_fd, v_prbp)

    plt.imshow(
        dImage_droughness_atprbp.detach().cpu().flip(0).numpy(),
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    plt.title("Gradient of Image w.r.t. Density atprbp")
    plt.axis("off")
    plt.show()

    return

    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(
        dImage_droughness_fd.detach().cpu().flip(0).numpy(),
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    axs[0].set_title("Gradient of Image w.r.t. Density fd")
    axs[0].axis("off")

    # axs[0, 1].imshow(
    #     dImage_droughness_rbp.detach().cpu().flip(0).numpy(),
    #     cmap="RdBu_r",
    #     vmin=-v,
    #     vmax=v,
    # )
    # axs[0, 1].set_title("Gradient of Image w.r.t. Roughness rbp")
    # axs[0, 1].axis("off")

    axs[1].imshow(
        dImage_droughness_prbp.detach().cpu().flip(0).numpy(),
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    axs[1].set_title("Gradient of Image w.r.t. Density prbp")
    axs[1].axis("off")

    axs[2].imshow(
        dImage_droughness_atprbp.detach().cpu().flip(0).numpy(),
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    axs[2].set_title("Gradient of Image w.r.t. Density atprbp")
    axs[2].axis("off")

    # plt.colorbar()
    # plt.tight_layout()
    plt.show()


main()
