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

n = 512


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


def estimate_gradient_fd(camera, scene, context, width, height):

    original_pos = camera.getPosition().numpy()

    h = 0.001
    offsets = [
        np.array([h, 0.0, 0.0], dtype=np.float32),
        np.array([-h, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, h, 0.0], dtype=np.float32),
        np.array([0.0, -h, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, h], dtype=np.float32),
        np.array([0.0, 0.0, -h], dtype=np.float32),
    ]

    imgs = []
    for offset in offsets:
        camera.setPosition(atcg.vec3(original_pos + offset))
        scene.setCamera(camera)

        dictionary = atcg.Dictionary()
        integrator = atcg.IntegratorRegistry.createIntegrator(
            "AttachedDiffPathtracingIntegrator",
            context,
            scene,
            width,
            height,
            dictionary,
        )

        img, _ = render_scene(integrator, 0, 1024 * n)
        imgs.append(img)

    camera.setPosition(atcg.vec3(original_pos))
    scene.setCamera(camera)

    dLdx = (imgs[1] - imgs[0]) / (2 * h)
    dLdy = (imgs[3] - imgs[2]) / (2 * h)
    dLdz = (imgs[5] - imgs[4]) / (2 * h)

    return dLdx[..., 0], dLdy[..., 0], dLdz[..., 0]


def estimate_gradient_atprbp(integrator: atcg.Integrator):
    num_samples = n * 1024
    rng_index = 0
    JL_estimate = 0
    for _ in tqdm(range(num_samples)):
        dictionary = atcg.Dictionary()
        dictionary.setUInt32("rng_index", rng_index)
        _, current_JL = integrator.forwardTrace(dictionary)
        current_JL.nan_to_num(0.0)
        JL_estimate += current_JL
        rng_index += 1

    return JL_estimate / num_samples


def main():

    width = 128
    height = 128

    props = atcg.WindowProps()
    props.width = width
    props.height = height
    props.hidden = True

    app = atcg.PythonApplication(props)
    atcg.PluginManager.loadPlugin("./bin/RelWithDebInfo/AttachedDiffPath.dll")

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
    integrator_att = atcg.IntegratorRegistry.createIntegrator(
        "AttachedDiffPathtracingIntegrator", context, scene, width, height, dictionary
    )

    JL_at = estimate_gradient_atprbp(integrator_att)

    JL_at = JL_at.detach().cpu().flip(0).numpy()
    JL_at = JL_at[..., ::3]

    _, axs = plt.subplots(2, 3)

    v = 1

    axs[0, 0].imshow(
        JL_at[..., 0],
        cmap="RdBu_r",
        vmin=-v / 2,
        vmax=v / 2,
    )
    axs[0, 0].set_title("dL/dx0.x")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(
        JL_at[..., 1],
        cmap="RdBu_r",
        vmin=-v / 2,
        vmax=v / 2,
    )
    axs[0, 1].set_title("dL/dx0.y")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(
        JL_at[..., 2],
        cmap="RdBu_r",
        vmin=-v / 2,
        vmax=v / 2,
    )
    axs[0, 2].set_title("dL/dx0.z")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(
        JL_at[..., 3],
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    axs[1, 0].set_title("dL/dx1.x")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(
        JL_at[..., 4],
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    axs[1, 1].set_title("dL/dx1.y")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(
        JL_at[..., 5],
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
    )
    axs[1, 2].set_title("dL/dx1.z")
    axs[1, 2].axis("off")

    plt.show()
    return
    dLdx, dLdy, dLdz = estimate_gradient_fd(camera, scene, context, width, height)

    dLdx = dLdx.detach().cpu().flip(0).numpy()
    dLdy = dLdy.detach().cpu().flip(0).numpy()
    dLdz = dLdz.detach().cpu().flip(0).numpy()

    _, axs = plt.subplots(1, 3)

    v = 1

    axs[0].imshow(
        dLdx,
        cmap="RdBu_r",
        vmin=-v / 2,
        vmax=v / 2,
    )
    axs[0].set_title("dL/dx0.x")
    axs[0].axis("off")

    axs[1].imshow(
        dLdy,
        cmap="RdBu_r",
        vmin=-v / 2,
        vmax=v / 2,
    )
    axs[1].set_title("dL/dx0.y")
    axs[1].axis("off")

    axs[2].imshow(
        dLdz,
        cmap="RdBu_r",
        vmin=-v / 2,
        vmax=v / 2,
    )
    axs[2].set_title("dL/dx0.z")
    axs[2].axis("off")

    plt.show()


main()
