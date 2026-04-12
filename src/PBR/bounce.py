import torch
import numpy as np
import pyatcg as atcg


class bounce(atcg.Behavior):

    def onAttach(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

        self.start = True

        transform = self.entity.getTransformComponent()
        self.position = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        transform.setPosition(atcg.vec3(self.position))
        self.entity.replaceTransformComponent(transform)

    def onUpdate(self, dt: float):
        if not self.start:
            return

        self.velocity += dt * np.array([0.0, -9.81, 0.0], dtype=np.float32)
        self.position += dt * self.velocity

        if self.position[1] < 0.0:
            # Estimate time of collision during the timestep
            t_hit = (
                dt
                * (self.position[1] - 0.0)
                / (self.position[1] - (self.position[1] - self.velocity[1] * dt) + 1e-5)
            )

            # Backtrack to self.position at collision
            self.position[1] -= self.velocity[1] * (dt - t_hit)

            # Reflect self.velocity
            self.velocity[1] = -self.velocity[1]

            # Forward integrate the rest of the timestep after bounce
            self.position[1] += self.velocity[1] * (dt - t_hit)

        transform = self.entity.getTransformComponent()
        transform.setPosition(atcg.vec3(self.position))
        self.entity.replaceTransformComponent(transform)

    def onEvent(self, event: atcg.Event):
        if event.getName() == "KeyPressed":
            if event.getKeyCode() == 66:  # B
                self.start = not self.start

                if self.start:
                    self.velocity = np.zeros(3, dtype=np.float32)
                    transform = self.entity.getTransformComponent()
                    self.position = np.array([0.0, 5.0, 0.0], dtype=np.float32)
                    transform.setPosition(atcg.vec3(self.position))
                    self.entity.replaceTransformComponent(transform)

    def onDetach(self):
        transform = self.entity.getTransformComponent()
        transform.setPosition(atcg.vec3(0, 0, 0))
        self.entity.replaceTransformComponent(transform)
