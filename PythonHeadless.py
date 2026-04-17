# RUN: pip install -e . before running this file

import torch
import numpy as np
import matplotlib.pyplot as plt
import platform

import os

os.environ["ATCG_BUILD_TYPE"] = "RelWithDebInfo"

if torch.cuda.is_available():
    os.environ["ATCG_CUDA_BACKEND"] = "On"
else:
    os.environ["ATCG_CUDA_BACKEND"] = "Off"

if torch.cuda.is_available() and "Linux" in platform.platform():
    os.environ["ATCG_HEADLESS"] = "On"

import pyatcg as atcg

def main():

    width = 1024
    height = 1024

    props = atcg.WindowProps()
    props.width = width
    props.height = height
    props.hidden = True

    app = atcg.PythonApplication(props)

    scene = atcg.read_scene(f"{atcg.resource_directory()}/test_scene.obj")

    skybox = atcg.Texture2D.create(atcg.imread(f"{atcg.resource_directory()}/pbr/skybox.hdr", 1.0))
    scene.setSkybox(skybox)

    extrinsics = atcg.CameraExtrinsics()
    extrinsics.setPosition(atcg.vec3(7., 3., 7.))
    extrinsics.setTarget(atcg.vec3(0.,0.,0.))
    instrinsics = atcg.CameraIntrinsics()
    camera = atcg.PerspectiveCamera(extrinsics, instrinsics)
    scene.setCamera(camera)

    framebuffer = atcg.Framebuffer(width, height)
    framebuffer.attachColor()
    framebuffer.attachDepth()
    framebuffer.complete()

    scene.draw(framebuffer)

    img = framebuffer.getColorAttachement(0).getData(torch.device("cpu"), 0)

    plt.imshow(img.flip(0))
    plt.axis("off")
    plt.show()

    if not torch.cuda.is_available():
        return

    context = atcg.RaytracingContextManager.createContext(0)

    integrator = atcg.PathIntegrator(context, scene, width, height)

    for _ in range(1024):
        render = integrator.generateRays()

    plt.imshow(render.cpu().flip(0))
    plt.axis("off")
    plt.show()

    _, ax = plt.subplots(1, 2)

    ax[0].imshow(img.flip(0))
    ax[0].axis("off")

    ax[1].imshow(render.cpu().flip(0))
    ax[1].axis("off")

    plt.show()


main()