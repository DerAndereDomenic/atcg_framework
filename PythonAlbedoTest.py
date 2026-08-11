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

n = 1


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

    bsdf = objects[0].getBSDFComponent().bsdf

    bsdf.setParameter(
        "diffuse_texture",
        torch.tensor([[[0.8, 0.5, 0.3]]], dtype=torch.float32, device="cuda"),
    )
    albedo = bsdf.getParameter("diffuse_texture")

    h = 0.001
    channels = []
    for i in range(3):
        albedo_new = albedo + h * torch.eye(3, device="cuda")[i].view(1, 1, 3)
        bsdf.setParameter("diffuse_texture", albedo_new)
        target_p, _ = render_scene(integrator, 0, 1024 * n)
        albedo_new = albedo - h * torch.eye(3, device="cuda")[i].view(1, 1, 3)
        bsdf.setParameter("diffuse_texture", albedo_new)
        target_m, _ = render_scene(integrator, 0, 1024 * n)
        # plt.imshow(target_m.detach().cpu().flip(0).numpy())
        # plt.show()
        target_p = torch.sum(target_p, dim=-1)
        target_m = torch.sum(target_m, dim=-1)

        dImage_dchannel = (target_p - target_m) / (2 * h)
        channels.append(dImage_dchannel)

    # plt.imshow(target_p.detach().cpu().flip(0).numpy())
    # plt.show()

    # Restore
    bsdf.setParameter("diffuse_texture", albedo)

    return torch.stack(channels)


def estimate_gradient_prbp(integrator: atcg.Integrator):
    print("Detached Path Replay Backpropagation")
    optix_scene = integrator.getOptixScene()

    objects = optix_scene.getEntitiesByName("sphere")

    bsdf = objects[0].getBSDFComponent().bsdf

    bsdf.setParameter(
        "diffuse_texture",
        torch.tensor([[[0.8, 0.5, 0.3]]], dtype=torch.float32, device="cuda"),
    )
    albedo = bsdf.getParameter("diffuse_texture")
    albedo_new = albedo.clone().requires_grad_(True)
    bsdf.setParameter("diffuse_texture", albedo_new)

    num_samples = 1024 * n
    rng_index = 0
    dImage_dalbedo_red = 0
    dImage_dalbedo_green = 0
    dImage_dalbedo_blue = 0
    for _ in tqdm(range(num_samples)):
        dictionary = atcg.Dictionary()
        dictionary.setUInt32("rng_index", rng_index)
        current_sample = integrator.forwardTrace(dictionary)
        rng_index += 1

        dictionary.setTensor("adjoint_y", torch.ones_like(current_sample))
        dictionary.setTensor("current_sample", current_sample)

        integrator.backwardTrace(dictionary)

        dImage_dalbedo_red += integrator.getAOVBuffer(0)
        dImage_dalbedo_green += integrator.getAOVBuffer(1)
        dImage_dalbedo_blue += integrator.getAOVBuffer(2)

    return (
        torch.stack([dImage_dalbedo_red, dImage_dalbedo_green, dImage_dalbedo_blue])
        / num_samples
    )


def estimate_gradient_atprbp(integrator: atcg.Integrator):
    print("Attached Path Replay Backpropagation")
    optix_scene = integrator.getOptixScene()

    objects = optix_scene.getEntitiesByName("sphere")

    bsdf = objects[0].getBSDFComponent().bsdf

    bsdf.setParameter(
        "diffuse_texture",
        torch.tensor([[[0.8, 0.5, 0.3]]], dtype=torch.float32, device="cuda"),
    )
    albedo = bsdf.getParameter("diffuse_texture")

    albedo_new = albedo.clone().requires_grad_(True)
    bsdf.setParameter("diffuse_texture", albedo_new)

    num_samples = 1024 * n
    rng_index = 0
    dImage_dalbedo_red = 0
    dImage_dalbedo_green = 0
    dImage_dalbedo_blue = 0
    for _ in tqdm(range(num_samples)):
        dictionary = atcg.Dictionary()
        dictionary.setUInt32("rng_index", rng_index)
        current_sample, current_JL = integrator.forwardTrace(dictionary)
        rng_index += 1

        dictionary.setTensor("adjoint_y", torch.ones_like(current_sample))
        dictionary.setTensor("current_sample", current_sample)
        dictionary.setTensor("JL_buffer", current_JL)

        integrator.backwardTrace(dictionary)

        dImage_dalbedo_red += integrator.getAOVBuffer(0)
        dImage_dalbedo_green += integrator.getAOVBuffer(1)
        dImage_dalbedo_blue += integrator.getAOVBuffer(2)

    return (
        torch.stack([dImage_dalbedo_red, dImage_dalbedo_green, dImage_dalbedo_blue])
        / num_samples
    )


def estimate_gradient_rbp(integrator: atcg.Integrator):
    print("Radiative Backpropagation")
    optix_scene = integrator.getOptixScene()

    objects = optix_scene.getEntitiesByName("sphere")

    bsdf = objects[0].getBSDFComponent().bsdf

    bsdf.setParameter(
        "diffuse_texture",
        torch.tensor([[[0.8, 0.5, 0.3]]], dtype=torch.float32, device="cuda"),
    )
    albedo = bsdf.getParameter("diffuse_texture")

    albedo_new = albedo.clone().requires_grad_(True)
    bsdf.setParameter("diffuse_texture", albedo_new)

    num_samples = 1024 * n
    rng_index = 0
    dImage_dalbedo_red = 0
    dImage_dalbedo_green = 0
    dImage_dalbedo_blue = 0
    for _ in tqdm(range(num_samples)):
        dictionary = atcg.Dictionary()
        dictionary.setUInt32("rng_index", rng_index)
        current_sample = integrator.forwardTrace(dictionary)
        rng_index += 1

        dictionary.setTensor("adjoint_y", torch.ones_like(current_sample))
        dictionary.setTensor("current_sample", current_sample)

        integrator.backwardTrace(dictionary)

        dImage_dalbedo_red += integrator.getAOVBuffer(0)
        dImage_dalbedo_green += integrator.getAOVBuffer(1)
        dImage_dalbedo_blue += integrator.getAOVBuffer(2)

    return (
        torch.stack([dImage_dalbedo_red, dImage_dalbedo_green, dImage_dalbedo_blue])
        / num_samples
    )


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

    context = atcg.RaytracingContextManager.createContext(0)

    atcg.Project.load("../DiffRendTestEdge/Project.json")
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
    dictionary.setUInt32("num_aovs", 3)
    integrator_det = atcg.IntegratorRegistry.createIntegrator(
        "DiffPathtracingIntegrator", context, scene, width, height, dictionary
    )

    integrator_att = atcg.IntegratorRegistry.createIntegrator(
        "AttachedDiffPathtracingIntegrator", context, scene, width, height, dictionary
    )

    integrator_rbp = atcg.IntegratorRegistry.createIntegrator(
        "RBPIntegrator", context, scene, width, height, dictionary
    )

    dImage_dalbedo_fd = estimate_gradient_fd(integrator_det)
    dImage_dalbedo_rbp = estimate_gradient_rbp(integrator_rbp)
    dImage_dalbedo_prbp = estimate_gradient_prbp(integrator_det)
    dImage_dalbedo_atprbp = estimate_gradient_atprbp(integrator_att)

    v = 4  # max(v_fd, v_prbp)

    for i in range(3):
        _, axs = plt.subplots(2, 2, figsize=(18, 6))

        axs[0, 0].imshow(
            dImage_dalbedo_fd[i].detach().cpu().flip(0).numpy(),
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        axs[0, 0].set_title("Gradient of Image w.r.t. Albedo fd")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(
            dImage_dalbedo_rbp[i].detach().cpu().flip(0).numpy(),
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        axs[0, 1].set_title("Gradient of Image w.r.t. Albedo rbp")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(
            dImage_dalbedo_prbp[i].detach().cpu().flip(0).numpy(),
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        axs[1, 0].set_title("Gradient of Image w.r.t. Albedo prbp")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(
            dImage_dalbedo_atprbp[i].detach().cpu().flip(0).numpy(),
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        axs[1, 1].set_title("Gradient of Image w.r.t. Albedo atprbp")
        axs[1, 1].axis("off")

        # plt.colorbar()
        plt.tight_layout()
        plt.show()


main()
