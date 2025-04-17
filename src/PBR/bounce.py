import torch
import numpy as np
import pyatcg as atcg

position = np.zeros(3, dtype=np.float32)
velocity = np.zeros(3, dtype=np.float32)

start = True


def onAttach(scene: atcg.Scene, ent: atcg.Entity):
    global position
    global velocity
    velocity = np.zeros(3, dtype=np.float32)
    transform = ent.getTransformComponent()
    position = np.array([0.0, 5.0, 0.0], dtype=np.float32)
    transform.setPosition(atcg.vec3(position))
    ent.replaceTransformComponent(transform)


def onUpdate(dt: float, scene: atcg.Scene, ent: atcg.Entity):
    global position
    global velocity
    global start

    if not start:
        return

    velocity += dt * np.array([0.0, -9.81, 0.0], dtype=np.float32)
    position += dt * velocity

    if position[1] < 0.0:
        # Estimate time of collision during the timestep
        t_hit = (
            dt
            * (position[1] - 0.0)
            / (position[1] - (position[1] - velocity[1] * dt) + 1e-5)
        )

        # Backtrack to position at collision
        position[1] -= velocity[1] * (dt - t_hit)

        # Reflect velocity
        velocity[1] = -velocity[1]

        # Forward integrate the rest of the timestep after bounce
        position[1] += velocity[1] * (dt - t_hit)

    transform = ent.getTransformComponent()
    transform.setPosition(atcg.vec3(position))
    ent.replaceTransformComponent(transform)


def onEvent(event: atcg.Event, scene: atcg.Scene, ent: atcg.Entity):
    global start
    global velocity
    global position
    if event.getName() == "KeyPressed":
        if event.getKeyCode() == 66:  # B
            start = not start

            if start:
                velocity = np.zeros(3, dtype=np.float32)
                transform = ent.getTransformComponent()
                position = np.array([0.0, 5.0, 0.0], dtype=np.float32)
                transform.setPosition(atcg.vec3(position))
                ent.replaceTransformComponent(transform)


def onDetach(scene: atcg.Scene, ent: atcg.Entity):
    transform = ent.getTransformComponent()
    transform.setPosition(atcg.vec3(0, 0, 0))
    ent.replaceTransformComponent(transform)
