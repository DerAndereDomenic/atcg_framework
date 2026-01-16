import torch
from torch.autograd.functional import jacobian

import torch
import copy


class TorchRNG:
    def __init__(self, seed=None, device="cpu"):
        self.device = device
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        else:
            # Randomize seed from PyTorch’s global RNG
            seed = torch.seed()
            self.generator.manual_seed(seed)
        self.seed = seed

    def random(self):
        """Return a random float in [0,1)."""
        return torch.rand((), generator=self.generator, device=self.device)

    def randint(self, low, high):
        """Integer in [low, high)."""
        return torch.randint(
            low, high, (1,), generator=self.generator, device=self.device
        )

    def normal(self, mean=0.0, std=1.0):
        """Normal distribution."""
        return torch.normal(
            mean, std, (1,), generator=self.generator, device=self.device
        )

    def get_state(self):
        """Return a copy of the RNG state."""
        return copy.deepcopy(self.generator.get_state())

    def set_state(self, state):
        """Restore state."""
        self.generator.set_state(state)

    def clone(self):
        """Deep copy including RNG state."""
        return copy.deepcopy(self)


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


class Plane:
    def __init__(self, point, normal):
        self.point = point
        self.normal = normal / torch.norm(normal)

    def intersect(self, ray):
        denom = torch.dot(self.normal, ray.direction)
        if torch.abs(denom) < 1e-6:
            return None  # No intersection, the ray is parallel to the plane
        d = torch.dot(self.point - ray.origin, self.normal) / denom
        if d < 0:
            return None  # The intersection is behind the ray's origin
        intersection_point = ray.origin + d * ray.direction
        return intersection_point

    def getNormal(self, x):
        return self.normal


class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, ray):
        L = self.center - ray.origin
        tca = torch.dot(L, ray.direction)
        d2 = torch.dot(L, L) - tca * tca
        radius2 = self.radius * self.radius
        if d2 > radius2:
            return None  # No intersection
        thc = torch.sqrt(radius2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0 and t1 < 0:
            return None  # Both intersections are behind the ray's origin
        t = t0 if t0 > 0 else t1
        intersection_point = ray.origin + t * ray.direction
        return intersection_point

    def getNormal(self, x):
        normal = (x - self.center) / self.radius
        return normal


class BSDFSamplingResult:

    def __init__(self, direction=None, pdf=0.0, brdf=None):
        self.direction = direction
        self.pdf = pdf
        self.brdf = brdf


class BSDF:
    def __init__(self, albedo, metallic, roughness):
        self.albedo = albedo
        self.metallic = metallic
        self.roughness = roughness

    def compute_local_frame(localZ):
        x = localZ[0]
        y = localZ[1]
        z = localZ[2]
        sz = 1 if (z >= 0) else -1
        a = 1 / (sz + z)
        ya = y * a
        b = x * ya
        c = x * sz

        localX = torch.zeros(3)
        localY = torch.zeros(3)

        localX[0] = c * x * a - 1
        localX[1] = sz * b
        localX[2] = c

        localY[0] = b
        localY[1] = y * ya - sz
        localY[2] = y

        frame = torch.zeros((3, 3))

        frame[:, 0] = localX
        frame[:, 1] = localY
        frame[:, 2] = localZ
        return frame

    def sample_hemisphere_cosine_weighted(rng):
        uv_x = rng.random()
        uv_y = rng.random()
        r = torch.sqrt(uv_x)
        phi = 2.0 * torch.pi * uv_y

        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt(torch.clamp(1 - uv_x, 0.0))

        dir = torch.zeros(3)

        dir[0] = x
        dir[1] = y
        dir[2] = z

        return dir

    def sample_ggx_specular(roughness, rng):
        uv_x = rng.random()
        uv_y = rng.random()
        cos_theta = torch.sqrt(
            (1.0 - uv_x) / (1.0 + (roughness * roughness - 1.0) * uv_x)
        )
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, 0.0))
        phi = 2.0 * torch.pi * uv_y

        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = cos_theta

        dir = torch.zeros(3)

        dir[0] = x
        dir[1] = y
        dir[2] = z

        return dir

    def reflect(I, N):
        return I - N * torch.dot(N, I) * 2.0

    def D_GGX(NdotH, roughness):
        a2 = roughness * roughness
        d = (NdotH * a2 - NdotH) * NdotH + 1.0
        return a2 / (torch.pi * d * d + 1e-5)

    def V_SmithGGX(NdotL, NdotV, alpha, eps=1e-8):
        a2 = alpha * alpha
        lambdaV = NdotL * torch.sqrt(NdotV * NdotV * (1.0 - a2) + a2)
        lambdaL = NdotV * torch.sqrt(NdotL * NdotL * (1.0 - a2) + a2)
        return 0.5 / (lambdaV + lambdaL + eps)

    def fresnel_schlick(F0, VdotH):
        return F0 + (torch.tensor([1.0, 1.0, 1.0], device=F0.device) - F0) * torch.pow(
            torch.clamp(1.0 - VdotH, min=0.0), 5.0
        )

    def warp_normal_to_reflected_direction_pdf(reflected_dir, normal):
        return 1.0 / torch.abs(4.0 * torch.dot(reflected_dir, normal))

    def sample(self, incoming_ray, object_normal, rng):
        sampled_dir = self.sample_direction(incoming_ray, object_normal, rng)
        brdf, pdf = self.eval(incoming_ray, sampled_dir, object_normal)

        return BSDFSamplingResult(direction=sampled_dir, pdf=pdf, brdf=brdf)

    def sample_direction(self, incoming_ray, object_normal, rng):
        roughness = torch.clamp(self.roughness**2, 1e-3)
        metallic_color = torch.lerp(
            torch.tensor([0.04, 0.04, 0.04]), self.albedo, self.metallic
        )
        diffuse_color = self.albedo * (1 - self.metallic)

        view_dir = -incoming_ray.direction
        normal = object_normal

        NdotV = torch.dot(normal, view_dir)
        if NdotV <= 0:
            print("NdotV <= 0")
            return BSDFSamplingResult()

        local_frame = BSDF.compute_local_frame(normal)

        diffuse_probability = torch.sum(diffuse_color) / (
            torch.sum(diffuse_color) + torch.sum(metallic_color)
        )

        if rng.random() < diffuse_probability:
            # Sample diffuse reflection
            sampled_dir = BSDF.sample_hemisphere_cosine_weighted(rng)
        else:
            # Sample specular reflection
            print("ggx sampling")
            local_halfway = BSDF.sample_ggx_specular(roughness, rng)
            halfway = local_frame @ local_halfway
            sampled_dir = BSDF.reflect(incoming_ray.direction, halfway)

        return sampled_dir

    def eval(self, incoming_ray, sampled_dir, object_normal):
        roughness = torch.clamp(self.roughness**2, 1e-3)
        metallic_color = torch.lerp(
            torch.tensor([0.04, 0.04, 0.04]), self.albedo, self.metallic
        )
        diffuse_color = self.albedo * (1 - self.metallic)

        view_dir = -incoming_ray.direction
        normal = object_normal

        NdotV = torch.dot(normal, view_dir)
        if NdotV <= 0:
            print("NdotV <= 0")
            return 0, 0

        diffuse_probability = torch.sum(diffuse_color) / (
            torch.sum(diffuse_color) + torch.sum(metallic_color)
        )

        specular_probability = 1.0 - diffuse_probability

        NdotL = torch.dot(normal, sampled_dir)
        if NdotL <= 0:
            print("NdotL <= 0")
            return 0, 0

        diffuse_bsdf = diffuse_color / torch.pi

        H = (sampled_dir + view_dir) / torch.norm(sampled_dir + view_dir)
        HdotV = torch.clamp(torch.dot(H, view_dir), 0.0, 1.0)
        NdotH = torch.clamp(torch.dot(normal, H), 0.0, 1.0)

        NDF = BSDF.D_GGX(NdotH, roughness)
        V = BSDF.V_SmithGGX(NdotL, NdotV, roughness)
        F = BSDF.fresnel_schlick(metallic_color, HdotV)

        kD = 1.0 - F

        specular_bsdf = NDF * V * F

        bsdf = (diffuse_bsdf * kD + specular_bsdf) * NdotL

        halfway_pdf = NDF * NdotH
        halfway_to_outgoing_pdf = BSDF.warp_normal_to_reflected_direction_pdf(
            sampled_dir, H
        )
        diffuse_pdf = NdotL / torch.pi
        specular_pdf = halfway_pdf * halfway_to_outgoing_pdf

        pdf = diffuse_probability * diffuse_pdf + specular_probability * specular_pdf

        return bsdf, pdf


class Emitter:
    def __init__(self, intensity, color):
        self.intensity = intensity
        self.color = color

    def evaluate(self):
        return self.intensity * self.color


class ShapeInstance:
    def __init__(self, shape=None, bsdf=None, emitter=None):
        self.shape = shape
        self.bsdf = bsdf
        self.emitter = emitter


def sample_L(plane1, plane2, bsdf, emitter, ray):
    throughput = torch.tensor([1.0, 1.0, 1.0])

    rng = TorchRNG(seed=42)

    current_ray = ray

    intersection = plane1.intersect(current_ray)

    bsdf_result = bsdf.sample(current_ray, plane1.getNormal(intersection), rng)

    throughput *= bsdf_result.brdf / bsdf_result.pdf

    current_ray = Ray(intersection, bsdf_result.direction)

    intersection2 = plane2.intersect(current_ray)

    bsdf_result = bsdf.sample(current_ray, plane2.getNormal(intersection2), rng)

    throughput *= bsdf_result.brdf / bsdf_result.pdf

    current_ray = Ray(intersection2, bsdf_result.direction)

    intersection3 = plane1.intersect(current_ray)

    Le = emitter.evaluate()

    # def test(x0, x1):
    #     throughput = torch.tensor([1.0, 1.0, 1.0])
    #     rng = TorchRNG(seed=42)

    #     # Iteration 1
    #     w = x1 - x0
    #     w = w / torch.norm(w)
    #     current_ray = Ray(x0, w)
    #     bsdf_result = bsdf.sample(current_ray, plane1.getNormal(x1), rng)

    #     # Spawn ray
    #     # current_ray = Ray(x1, bsdf_result.direction)
    #     # x2 = plane2.intersect(current_ray)

    #     throughput *= bsdf_result.brdf / bsdf_result.pdf

    #     # Iteration 2
    #     # x1 = x1.detach()
    #     # x2 = x2.detach()
    #     # w = x2 - x1
    #     # w = w / torch.norm(w)
    #     # current_ray = Ray(x1, w)
    #     # bsdf_result = bsdf.sample(current_ray, plane2.getNormal(x2), rng)

    #     # Spawn ray
    #     # current_ray = Ray(x2, bsdf_result.direction)
    #     # x3 = plane1.intersect(current_ray)

    #     # throughput *= bsdf_result.brdf / bsdf_result.pdf
    #     Le = emitter.evaluate()

    #     L = throughput * Le

    #     # print("Sanity", L)

    #     return L

    # Jt = jacobian(test, (ray.origin, intersection))
    # frame0 = BSDF.compute_local_frame(ray.direction)[:, :2]
    # frame1 = BSDF.compute_local_frame(plane1.getNormal(intersection))[:, :2]
    # Jt = torch.cat([Jt[0] @ frame0, Jt[1] @ frame1], dim=1)
    # print("Jt", Jt)

    return throughput * Le


def forward_pass(plane1, plane2, bsdf, emitter, ray):

    rng = TorchRNG(seed=42)

    beta = torch.ones(3)
    JL = torch.zeros(3, 6)
    Jbeta = torch.zeros(3, 6)
    Jray = torch.eye(6)

    current_ray = ray

    intersection = plane1.intersect(current_ray)
    # bsdf_result = scene[0].bsdf.sample(current_ray, scene[0].shape.normal)

    def intersect_plane1_fn(origin, direction):
        temp_ray = Ray(origin, direction)
        return plane1.intersect(temp_ray)

    Jintersect = jacobian(
        intersect_plane1_fn, (current_ray.origin, current_ray.direction)
    )

    def bsdf_plane1_fn(origin, direction):
        global bsdf_result
        temp_ray = Ray(origin, direction)
        intersection = plane1.intersect(temp_ray)
        sampled_dir = bsdf.sample_direction(
            temp_ray, plane1.getNormal(intersection), rng
        )
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane1.getNormal(intersection))
        bsdf_result = BSDFSamplingResult(brdf=brdf, pdf=pdf, direction=sampled_dir)
        return bsdf_result.brdf / bsdf_result.pdf, bsdf_result.direction

    Jsample = jacobian(bsdf_plane1_fn, (current_ray.origin, current_ray.direction))

    Jposition_ = torch.cat([Jintersect[0], Jintersect[1]], dim=1)
    Jdirection_ = torch.cat([Jsample[1][0], Jsample[1][1]], dim=1)

    Jray_ = torch.cat([Jposition_, Jdirection_], dim=0)

    # Jbsdf = torch.cat([Jsample[0][0], Jsample[0][1]], dim=1)

    # Jbsdf = Jbsdf @ Jray

    # Jray = Jray_ @ Jray

    # JL += 0

    # Jbeta = (
    #     torch.diag(bsdf_result.brdf / bsdf_result.pdf) @ Jbeta
    #     + torch.diag(beta) @ Jbsdf
    # )
    beta *= bsdf_result.brdf / bsdf_result.pdf

    current_ray = Ray(intersection.detach(), bsdf_result.direction.detach())

    intersection = plane2.intersect(current_ray)
    # bsdf_result = scene[0].bsdf.sample(current_ray, scene[0].shape.normal)

    def intersect_plane2_fn(origin, direction):
        temp_ray = Ray(origin, direction)
        return plane2.intersect(temp_ray)

    Jintersect = jacobian(
        intersect_plane2_fn, (current_ray.origin, current_ray.direction)
    )

    def bsdf_plane2_fn(origin, direction):
        global bsdf_result
        temp_ray = Ray(origin, direction)
        intersection = plane2.intersect(temp_ray)
        sampled_dir = bsdf.sample_direction(
            temp_ray, plane2.getNormal(intersection), rng
        )
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane2.getNormal(intersection))
        bsdf_result = BSDFSamplingResult(brdf=brdf, pdf=pdf, direction=sampled_dir)
        return bsdf_result.brdf / bsdf_result.pdf, bsdf_result.direction

    Jsample = jacobian(bsdf_plane2_fn, (current_ray.origin, current_ray.direction))

    Jposition_ = torch.cat([Jintersect[0], Jintersect[1]], dim=1)
    Jdirection_ = torch.cat([Jsample[1][0], Jsample[1][1]], dim=1)

    Jray_ = torch.cat([Jposition_, Jdirection_], dim=0)

    Jbsdf = torch.cat([Jsample[0][0], Jsample[0][1]], dim=1)

    Jbsdf = Jbsdf @ Jray

    Jray = Jray_ @ Jray

    # JL += 0

    Jbeta = (
        torch.diag(bsdf_result.brdf / bsdf_result.pdf) @ Jbeta
        + torch.diag(beta) @ Jbsdf
    )
    beta *= bsdf_result.brdf / bsdf_result.pdf

    current_ray = Ray(intersection, bsdf_result.direction)

    intersection2 = plane1.intersect(current_ray)

    Le = emitter.evaluate()
    L = beta * Le

    JLe = torch.zeros((3, 6))  # effectively independent on the incoming ray

    JL += torch.diag(beta) @ JLe + torch.diag(Le) @ Jbeta

    return L, JL


def backward_pass(plane1, plane2, bsdf, emitter, ray, L, JL):
    rng = TorchRNG(seed=42)

    beta = torch.ones(3)
    Jray = torch.eye(6)
    JL_albedo = torch.zeros((3, 3))
    JL_metallic = torch.zeros((3, 1))
    JL_roughness = torch.zeros((3, 1))

    current_ray = ray

    intersection = plane1.intersect(current_ray)
    # bsdf_result = scene[0].bsdf.sample(current_ray, scene[0].shape.normal)

    def intersect_plane1_fn(origin, direction):
        temp_ray = Ray(origin, direction)
        return plane1.intersect(temp_ray)

    def bsdf_plane1_fn(origin, direction):
        global bsdf_result
        temp_ray = Ray(origin, direction)
        intersection = plane1.intersect(temp_ray)
        sampled_dir = bsdf.sample_direction(
            temp_ray, plane1.getNormal(intersection), rng
        )
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane1.getNormal(intersection))
        bsdf_result = BSDFSamplingResult(brdf=brdf, pdf=pdf, direction=sampled_dir)
        return bsdf_result.brdf / bsdf_result.pdf, bsdf_result.direction

    # Jintersect = jacobian(
    #     intersect_plane1_fn, (current_ray.origin, current_ray.direction)
    # )

    rng_copy = rng.clone()
    Jsample = jacobian(bsdf_plane1_fn, (current_ray.origin, current_ray.direction))

    # Jposition_ = torch.cat([Jintersect[0], Jintersect[1]], dim=1)
    # Jdirection_ = torch.cat([Jsample[1][0], Jsample[1][1]], dim=1)

    # Jray_ = torch.cat([Jposition_, Jdirection_], dim=0)

    # Jbsdf = torch.cat([Jsample[0][0], Jsample[0][1]], dim=1)

    # Jbsdf = Jbsdf @ Jray

    # Jray = Jray_ @ Jray

    # JL -= torch.diag(L / (bsdf_result.brdf / bsdf_result.pdf)) @ Jbsdf
    # JL_ = JL[:, 3:] @ torch.linalg.inv(Jray[3:, 3:])
    JL_ = (JL @ torch.linalg.pinv(Jray))[:, 3:]

    # def bsdf_sample_bw(albedo, metallic, roughness, origin, direction):
    #     temp_ray = Ray(origin, direction)
    #     scene[0].bsdf.albedo = albedo
    #     scene[0].bsdf.metallic = metallic
    #     scene[0].bsdf.roughness = roughness
    #     sampled_dir = scene[0].bsdf.sample_direction(
    #         temp_ray, scene[0].shape.normal, rng_copy
    #     )
    #     return sampled_dir

    def bsdf_eval_plane1_bw(albedo, metallic, roughness, origin, direction):
        global bsdf_result
        temp_ray = Ray(origin, direction)
        bsdf.albedo = albedo
        bsdf.metallic = metallic
        bsdf.roughness = roughness
        sampled_dir = bsdf.sample_direction(
            temp_ray, plane1.getNormal(intersection), rng_copy
        )
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane1.getNormal(intersection))
        return brdf / pdf, sampled_dir

    Jbsdf_bw, Jwo_bw = jacobian(
        bsdf_eval_plane1_bw,
        (
            bsdf.albedo,
            bsdf.metallic,
            bsdf.roughness,
            current_ray.origin,
            current_ray.direction,
            # bsdf_result.direction.detach(),
        ),
    )

    # Jwo_bw = jacobian(
    #     bsdf_sample_bw,
    #     (
    #         scene[0].bsdf.albedo,
    #         scene[0].bsdf.metallic,
    #         scene[0].bsdf.roughness,
    #         current_ray.origin,
    #         current_ray.direction,
    #     ),
    # )

    Jbsdf_albedo = Jbsdf_bw[0]
    Jbsdf_metallic = Jbsdf_bw[1]
    Jbsdf_roughness = Jbsdf_bw[2]

    Jwo_albedo = Jwo_bw[0]
    Jwo_metallic = Jwo_bw[1]
    Jwo_roughness = Jwo_bw[2]

    dL_bsdf = L / (bsdf_result.brdf / bsdf_result.pdf)

    JL_albedo += torch.diag(dL_bsdf) @ Jbsdf_albedo
    JL_metallic += torch.diag(dL_bsdf) @ Jbsdf_metallic
    JL_roughness += torch.diag(dL_bsdf) @ Jbsdf_roughness

    JL_wo = JL_  # [:, 3:]

    JL_albedo += JL_wo @ Jwo_albedo
    JL_metallic += JL_wo @ Jwo_metallic
    JL_roughness += JL_wo @ Jwo_roughness

    beta *= bsdf_result.brdf / bsdf_result.pdf

    current_ray = Ray(intersection, bsdf_result.direction)

    intersection = plane2.intersect(current_ray)
    # bsdf_result = scene[0].bsdf.sample(current_ray, scene[0].shape.normal)

    def intersect_plane2_fn(origin, direction):
        temp_ray = Ray(origin, direction)
        return plane2.intersect(temp_ray)

    def bsdf_plane2_fn(origin, direction):
        global bsdf_result
        temp_ray = Ray(origin, direction)
        intersection = plane2.intersect(temp_ray)
        sampled_dir = bsdf.sample_direction(
            temp_ray, plane2.getNormal(intersection), rng
        )
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane2.getNormal(intersection))
        bsdf_result = BSDFSamplingResult(brdf=brdf, pdf=pdf, direction=sampled_dir)
        return bsdf_result.brdf / bsdf_result.pdf, bsdf_result.direction

    Jintersect = jacobian(
        intersect_plane2_fn, (current_ray.origin, current_ray.direction)
    )

    rng_copy = rng.clone()
    Jsample = jacobian(bsdf_plane2_fn, (current_ray.origin, current_ray.direction))

    Jposition_ = torch.cat([Jintersect[0], Jintersect[1]], dim=1)
    Jdirection_ = torch.cat([Jsample[1][0], Jsample[1][1]], dim=1)

    Jray_ = torch.cat([Jposition_, Jdirection_], dim=0)

    Jbsdf = torch.cat([Jsample[0][0], Jsample[0][1]], dim=1)

    Jbsdf = Jbsdf @ Jray

    Jray = Jray_ @ Jray

    JL -= torch.diag(L / (bsdf_result.brdf / bsdf_result.pdf)) @ Jbsdf
    # JL_ = JL[:, 3:] @ torch.linalg.inv(Jray[3:, 3:])
    JL_ = (JL @ torch.linalg.pinv(Jray))[:, 3:]

    # def bsdf_sample_bw(albedo, metallic, roughness, origin, direction):
    #     temp_ray = Ray(origin, direction)
    #     scene[0].bsdf.albedo = albedo
    #     scene[0].bsdf.metallic = metallic
    #     scene[0].bsdf.roughness = roughness
    #     sampled_dir = scene[0].bsdf.sample_direction(
    #         temp_ray, scene[0].shape.normal, rng_copy
    #     )
    #     return sampled_dir

    def bsdf_eval_plane2_bw(albedo, metallic, roughness, origin, direction):
        global bsdf_result
        temp_ray = Ray(origin, direction)
        bsdf.albedo = albedo
        bsdf.metallic = metallic
        bsdf.roughness = roughness
        sampled_dir = bsdf.sample_direction(
            temp_ray, plane2.getNormal(intersection), rng_copy
        )
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane2.getNormal(intersection))
        return brdf / pdf, sampled_dir

    Jbsdf_bw, Jwo_bw = jacobian(
        bsdf_eval_plane2_bw,
        (
            bsdf.albedo,
            bsdf.metallic,
            bsdf.roughness,
            current_ray.origin,
            current_ray.direction,
            # bsdf_result.direction.detach(),
        ),
    )

    # Jwo_bw = jacobian(
    #     bsdf_sample_bw,
    #     (
    #         scene[0].bsdf.albedo,
    #         scene[0].bsdf.metallic,
    #         scene[0].bsdf.roughness,
    #         current_ray.origin,
    #         current_ray.direction,
    #     ),
    # )

    Jbsdf_albedo = Jbsdf_bw[0]
    Jbsdf_metallic = Jbsdf_bw[1]
    Jbsdf_roughness = Jbsdf_bw[2]

    Jwo_albedo = Jwo_bw[0]
    Jwo_metallic = Jwo_bw[1]
    Jwo_roughness = Jwo_bw[2]

    dL_bsdf = L / (bsdf_result.brdf / bsdf_result.pdf)

    JL_albedo += torch.diag(dL_bsdf) @ Jbsdf_albedo
    JL_metallic += torch.diag(dL_bsdf) @ Jbsdf_metallic
    JL_roughness += torch.diag(dL_bsdf) @ Jbsdf_roughness

    JL_wo = JL_  # [:, 3:]

    JL_albedo += JL_wo @ Jwo_albedo
    JL_metallic += JL_wo @ Jwo_metallic
    JL_roughness += JL_wo @ Jwo_roughness

    beta *= bsdf_result.brdf / bsdf_result.pdf

    print(JL_albedo)
    print(JL_metallic)
    print(JL_roughness)

    # print(JL_[:, 3:])
    # print(JL_[:, :3] @ torch.inverse(Jray[3:, 3:]))
    # print(Jray[3:, 3:])

    # Jbeta = beta * Jbsdf
    # beta *= bsdf_result.brdf / bsdf_result.pdf

    # current_ray = Ray(intersection, bsdf_result.direction)

    # # beta *= bsdf_result.brdf / bsdf_result.pdf

    # # current_ray = Ray(intersection, bsdf_result.direction)

    # # # intersection2 = scene[1].shape.intersect(current_ray)
    # Le = scene[1].emitter.evaluate()
    # L = beta * Le

    # JLe = torch.zeros((3, 6))  # effectively independent on the incoming ray

    # JL += torch.diag(beta) @ JLe + torch.diag(Le) @ Jbsdf

    # return L, JL


def prbprob(plane1, plane2, bsdf, emitter, ray):
    L, JL = forward_pass(plane1, plane2, bsdf, emitter, ray)

    print(L, JL)

    backward_pass(plane1, plane2, bsdf, emitter, ray, L, JL)


def trace_forward(plane1, plane2, bsdf, emitter, ray):

    L = 0
    beta = torch.ones(3)
    JL = torch.zeros((3, 4))
    Jbeta = torch.zeros((3, 4))
    Jray = torch.eye(4)

    rng = TorchRNG(seed=42)

    # Iteration 1
    frame0 = BSDF.compute_local_frame(ray.direction)[:, :2]
    intersection1 = plane1.intersect(ray)
    frame1 = BSDF.compute_local_frame(plane1.getNormal(intersection1))[:, :2]

    def sample_ray1(p1, p2):
        global intersection2
        global bsdf_result
        w = p2 - p1
        w = w / torch.norm(w)
        temp_ray = Ray(p1, w)
        bsdf_result = bsdf.sample(temp_ray, plane1.getNormal(p2), rng)
        intersection2 = plane2.intersect(Ray(p2, bsdf_result.direction))

        return p2, intersection2, bsdf_result.brdf / bsdf_result.pdf

    Jsample = jacobian(sample_ray1, (ray.origin, intersection1))
    frame2 = BSDF.compute_local_frame(plane2.getNormal(intersection2))[:, :2]

    print("Jx1x0", Jsample[0][0])

    Jx1x0 = frame1.T @ Jsample[0][0] @ frame0
    Jx2x0 = frame2.T @ Jsample[1][0] @ frame0
    Jx1x1 = frame1.T @ Jsample[0][1] @ frame1
    Jx2x1 = frame2.T @ Jsample[1][1] @ frame1

    J1 = torch.cat([Jx1x0, Jx1x1], dim=1)
    J2 = torch.cat([Jx2x0, Jx2x1], dim=1)
    Jray_ = torch.cat([J1, J2], dim=0)
    Jbsdfx0 = Jsample[2][0] @ frame0
    Jbsdfx1 = Jsample[2][1] @ frame1
    Jbsdf = torch.cat([Jbsdfx0, Jbsdfx1], dim=1)
    # Jbsdf = Jbsdf @ Jray
    Jray = Jray_ @ Jray
    Jbeta = Jbsdf

    beta *= bsdf_result.brdf / bsdf_result.pdf

    # Le = emitter.evaluate()
    # L = beta * Le

    # JLe = torch.zeros((3, 4))  # effectively independent on the incoming ray
    # JL = torch.diag(beta) @ JLe + torch.diag(Le) @ Jbeta

    # print("L", L)
    # print("JL", JL)

    # return L, JL

    # Iteration 2
    frame0 = frame1
    # intersection3 = plane1.intersect(Ray(intersection2, bsdf_result.direction))
    frame1 = frame2

    def sample_ray2(p1, p2):
        global intersection3
        global bsdf_result
        w = p2 - p1
        w = w / torch.norm(w)
        temp_ray = Ray(p1, w)
        bsdf_result = bsdf.sample(temp_ray, plane2.getNormal(p2), rng)
        intersection3 = plane1.intersect(Ray(p2, bsdf_result.direction))

        return p2, intersection3, bsdf_result.brdf / bsdf_result.pdf

    Jsample = jacobian(sample_ray2, (intersection1, intersection2))
    frame2 = BSDF.compute_local_frame(plane1.getNormal(intersection3))[:, :2]

    print("Jx1x0", Jsample[0][0])
    Jx1x0 = frame1.T @ Jsample[0][0] @ frame0
    Jx2x0 = frame2.T @ Jsample[1][0] @ frame0
    Jx1x1 = frame1.T @ Jsample[0][1] @ frame1
    Jx2x1 = frame2.T @ Jsample[1][1] @ frame1

    J1 = torch.cat([Jx1x0, Jx1x1], dim=1)
    J2 = torch.cat([Jx2x0, Jx2x1], dim=1)
    Jray_ = torch.cat([J1, J2], dim=0)
    Jbsdfx0 = Jsample[2][0] @ frame0
    Jbsdfx1 = Jsample[2][1] @ frame1
    Jbsdf = torch.cat([Jbsdfx0, Jbsdfx1], dim=1)
    Jbsdf = Jbsdf @ Jray
    Jray = Jray_ @ Jray
    Jbeta = (
        torch.diag(bsdf_result.brdf / bsdf_result.pdf) @ Jbeta
        + torch.diag(beta) @ Jbsdf
    )
    beta *= bsdf_result.brdf / bsdf_result.pdf

    # Iteration 3 (just evaluate light source)
    Le = emitter.evaluate()
    L = beta * Le

    JLe = torch.zeros((3, 4))  # effectively independent on the incoming ray
    JL += torch.diag(beta) @ JLe + torch.diag(Le) @ Jbeta

    print("L", L)
    print("JL", JL)

    return L, JL


def trace_backward(plane1, plane2, bsdf, emitter, ray, L, JL):
    # return
    beta = torch.ones(3)
    Jray = torch.eye(4)

    JL_albedo = torch.zeros((3, 3))
    JL_metallic = torch.zeros((3, 1))
    JL_roughness = torch.zeros((3, 1))

    rng = TorchRNG(seed=42)

    # Iteration 1
    frame0 = BSDF.compute_local_frame(ray.direction)[:, :2]
    intersection1 = plane1.intersect(ray)
    frame1 = BSDF.compute_local_frame(plane1.getNormal(intersection1))[:, :2]

    def sample_ray1(p1, p2):
        global intersection2
        global bsdf_result
        w = p2 - p1
        w = w / torch.norm(w)
        temp_ray = Ray(p1, w)
        bsdf_result = bsdf.sample(temp_ray, plane1.getNormal(p2), rng)
        intersection2 = plane2.intersect(Ray(p2, bsdf_result.direction))

        return p2, intersection2, bsdf_result.brdf / bsdf_result.pdf

    rng_copy = rng.clone()
    Jsample = jacobian(sample_ray1, (ray.origin, intersection1))
    frame2 = BSDF.compute_local_frame(plane2.getNormal(intersection2))[:, :2]

    Jx1x0 = frame1.T @ Jsample[0][0] @ frame0
    Jx2x0 = frame2.T @ Jsample[1][0] @ frame0
    Jx1x1 = frame1.T @ Jsample[0][1] @ frame1
    Jx2x1 = frame2.T @ Jsample[1][1] @ frame1

    J1 = torch.cat([Jx1x0, Jx1x1], dim=1)
    J2 = torch.cat([Jx2x0, Jx2x1], dim=1)
    Jray_ = torch.cat([J1, J2], dim=0)
    Jbsdfx0 = Jsample[2][0] @ frame0
    Jbsdfx1 = Jsample[2][1] @ frame1
    Jbsdf = torch.cat([Jbsdfx0, Jbsdfx1], dim=1)
    Jbsdf = Jbsdf @ Jray
    Jray = Jray_ @ Jray

    JL -= torch.diag(L / (bsdf_result.brdf / bsdf_result.pdf)) @ Jbsdf
    JL_ = JL @ torch.linalg.inv(Jray)

    def bsdf_eval_plane1_bw(albedo, metallic, roughness, p1, p2):
        global bsdf_result
        w = p2 - p1
        w = w / torch.norm(w)
        temp_ray = Ray(p1, w)
        bsdf.albedo = albedo
        bsdf.metallic = metallic
        bsdf.roughness = roughness
        sampled_dir = bsdf.sample_direction(temp_ray, plane1.getNormal(p2), rng_copy)
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane1.getNormal(p2))
        return brdf / pdf, sampled_dir

    Jbsdf_bw, Jwo_bw = jacobian(
        bsdf_eval_plane1_bw,
        (
            bsdf.albedo,
            bsdf.metallic,
            bsdf.roughness,
            ray.origin,
            intersection1,
            # bsdf_result.direction.detach(),
        ),
    )

    def intersect1(w):
        return plane2.intersect(Ray(intersection1, w))

    (Jx2,) = jacobian(intersect1, (bsdf_result.direction.detach(),))

    Jx2_wo = frame2.T @ Jx2

    JL_wo = JL_[:, 2:] @ Jx2_wo

    Jbsdf_albedo = Jbsdf_bw[0]
    Jbsdf_metallic = Jbsdf_bw[1]
    Jbsdf_roughness = Jbsdf_bw[2]

    Jwo_albedo = Jwo_bw[0]
    Jwo_metallic = Jwo_bw[1]
    Jwo_roughness = Jwo_bw[2]

    dL_bsdf = L / (bsdf_result.brdf / bsdf_result.pdf)

    JL_albedo += torch.diag(dL_bsdf) @ Jbsdf_albedo
    JL_metallic += torch.diag(dL_bsdf) @ Jbsdf_metallic
    JL_roughness += torch.diag(dL_bsdf) @ Jbsdf_roughness

    JL_albedo += JL_wo @ Jwo_albedo
    JL_metallic += JL_wo @ Jwo_metallic
    JL_roughness += JL_wo @ Jwo_roughness

    beta *= bsdf_result.brdf / bsdf_result.pdf

    # Iteration 2
    frame0 = frame1
    # intersection3 = plane1.intersect(Ray(intersection2, bsdf_result.direction))
    frame1 = frame2

    def sample_ray2(p1, p2):
        global intersection3
        global bsdf_result
        w = p2 - p1
        w = w / torch.norm(w)
        temp_ray = Ray(p1, w)
        bsdf_result = bsdf.sample(temp_ray, plane2.getNormal(p2), rng)
        intersection3 = plane1.intersect(Ray(p2, bsdf_result.direction))

        return p2, intersection3, bsdf_result.brdf / bsdf_result.pdf

    rng_copy = rng.clone()
    Jsample = jacobian(sample_ray2, (intersection1, intersection2))
    frame2 = BSDF.compute_local_frame(plane1.getNormal(intersection3))[:, :2]

    Jx1x0 = frame1.T @ Jsample[0][0] @ frame0
    Jx2x0 = frame2.T @ Jsample[1][0] @ frame0
    Jx1x1 = frame1.T @ Jsample[0][1] @ frame1
    Jx2x1 = frame2.T @ Jsample[1][1] @ frame1

    J1 = torch.cat([Jx1x0, Jx1x1], dim=1)
    J2 = torch.cat([Jx2x0, Jx2x1], dim=1)
    Jray_ = torch.cat([J1, J2], dim=0)
    Jbsdfx0 = Jsample[2][0] @ frame0
    Jbsdfx1 = Jsample[2][1] @ frame1
    Jbsdf = torch.cat([Jbsdfx0, Jbsdfx1], dim=1)
    Jbsdf = Jbsdf @ Jray
    Jray = Jray_ @ Jray

    JL -= torch.diag(L / (bsdf_result.brdf / bsdf_result.pdf)) @ Jbsdf
    JL_ = JL @ torch.linalg.inv(Jray)

    def bsdf_eval_plane2_bw(albedo, metallic, roughness, p1, p2):
        global bsdf_result
        w = p2 - p1
        w = w / torch.norm(w)
        temp_ray = Ray(p1, w)
        bsdf.albedo = albedo
        bsdf.metallic = metallic
        bsdf.roughness = roughness
        sampled_dir = bsdf.sample_direction(temp_ray, plane2.getNormal(p2), rng_copy)
        brdf, pdf = bsdf.eval(temp_ray, sampled_dir, plane2.getNormal(p2))
        return brdf / pdf, sampled_dir

    Jbsdf_bw, Jwo_bw = jacobian(
        bsdf_eval_plane2_bw,
        (
            bsdf.albedo,
            bsdf.metallic,
            bsdf.roughness,
            intersection1,
            intersection2,
            # bsdf_result.direction.detach(),
        ),
    )

    def intersect2(w):
        return plane1.intersect(Ray(intersection2, w))

    (Jx2,) = jacobian(intersect2, (bsdf_result.direction.detach(),))

    Jx2_wo = frame2.T @ Jx2

    JL_wo = JL_[:, 2:] @ Jx2_wo

    Jbsdf_albedo = Jbsdf_bw[0]
    Jbsdf_metallic = Jbsdf_bw[1]
    Jbsdf_roughness = Jbsdf_bw[2]

    Jwo_albedo = Jwo_bw[0]
    Jwo_metallic = Jwo_bw[1]
    Jwo_roughness = Jwo_bw[2]

    dL_bsdf = L / (bsdf_result.brdf / bsdf_result.pdf)

    JL_albedo += torch.diag(dL_bsdf) @ Jbsdf_albedo
    JL_metallic += torch.diag(dL_bsdf) @ Jbsdf_metallic
    JL_roughness += torch.diag(dL_bsdf) @ Jbsdf_roughness

    JL_albedo += JL_wo @ Jwo_albedo
    JL_metallic += JL_wo @ Jwo_metallic
    JL_roughness += JL_wo @ Jwo_roughness

    print("albedo", JL_albedo)
    print("metallic", JL_metallic)
    print("roughness", JL_roughness)

    beta *= bsdf_result.brdf / bsdf_result.pdf

    return
    # Iteration 3 (just evaluate light source)
    Le = emitter.evaluate()
    L = beta * Le

    JLe = torch.zeros((3, 4))  # effectively independent on the incoming ray
    JL += torch.diag(beta) @ JLe + torch.diag(Le) @ Jbeta

    print(L, JL)

    return L, JL


def main():
    plane1 = Plane(torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))
    plane2 = Plane(torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))

    plane1 = Sphere(torch.tensor([0.0, 15.0, 0.0]), torch.tensor([14.0]))
    plane2 = Sphere(torch.tensor([0.0, -15.0, 0.0]), torch.tensor([14.0]))

    bsdf = BSDF(
        albedo=torch.tensor([0.8, 0.2, 0.2]),
        metallic=torch.tensor([0.5]),
        roughness=torch.tensor([0.3]),
    )

    emitter = Emitter(
        intensity=10.0,
        color=torch.tensor([1.0, 1.0, 1.0]),
    )

    shape_instance1 = ShapeInstance(shape=plane1, bsdf=bsdf, emitter=None)
    shape_instance2 = ShapeInstance(shape=plane2, bsdf=None, emitter=emitter)

    scene = [shape_instance1, shape_instance2]

    ray = Ray(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))

    L, JL = trace_forward(plane1, plane2, bsdf, emitter, ray)
    trace_backward(plane1, plane2, bsdf, emitter, ray, L, JL)

    # sample_L(plane1, plane2, bsdf, emitter, ray)

    # return

    p2 = plane1.intersect(ray)
    n = plane1.getNormal(p2)

    def compute_L(p1, p2):
        w = p2 - p1
        w = w / torch.norm(w)
        temp_ray = Ray(p1, w)
        L = sample_L(plane1, plane2, bsdf, emitter, temp_ray)
        print(L)
        return L

    frame0 = BSDF.compute_local_frame(ray.direction)[:, :2]
    frame1 = BSDF.compute_local_frame(n)[:, :2]
    JL = jacobian(compute_L, (ray.origin, p2))
    JL = torch.cat([JL[0] @ frame0, JL[1] @ frame1], dim=1)
    print(JL)
    # return

    def compute_L(albedo, metallic, roughness):
        scene[0].bsdf.albedo = albedo
        scene[0].bsdf.metallic = metallic
        scene[0].bsdf.roughness = roughness
        L = sample_L(plane1, plane2, bsdf, emitter, ray)
        print(L)
        return L

    params = (bsdf.albedo, bsdf.metallic, bsdf.roughness)

    JL = jacobian(compute_L, params)
    print(JL)

    # prbprob(plane1, plane2, bsdf, emitter, ray)


if __name__ == "__main__":
    main()
