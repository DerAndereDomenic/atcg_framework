#version 330 core

#define PI 3.14159

layout (location = 0) out vec4 outColor;

in vec3 frag_normal;
in vec3 frag_pos;
in vec3 frag_color;
in vec3 bbox_max;
in vec3 bbox_min;

uniform vec3 camera_pos;
uniform vec3 camera_dir;
uniform vec3 flat_color;
uniform float sigma_s_base;
uniform float sigma_a_base;
uniform float g;
uniform vec3 light_position;

uniform sampler3D noise_texture;

struct HitInfo
{
    bool hit;
    float tmin;
    float tmax;
};

vec3 light_intensity = vec3(1,1,1);
vec3 ambient = -log(1 - pow(vec3(76.0, 76.0, 128.0) / 255.0, vec3(2.4)));

float sigma_a(vec3 pos)
{
    return sigma_a_base;
}

float sigma_s(vec3 pos)
{
    float noise = texture(noise_texture, (pos+1)/2).r;
    return max(0.0, (0.2 - noise)) * sigma_s_base;
}

float sigma_t(vec3 pos)
{
    return sigma_s(pos) + sigma_a(pos);
}

vec3 radiance = vec3(0);

HitInfo intersect(vec3 ray_origin, vec3 ray_dir)
{
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    vec3 ray_invdir = 1.0 / ray_dir;

    vec3 bounds[2];
    bounds[0] = bbox_min;
    bounds[1] = bbox_max;

    int signs[3];
    signs[0] = ray_dir.x < 0 ? 1 : 0;
    signs[1] = ray_dir.y < 0 ? 1 : 0;
    signs[2] = ray_dir.z < 0 ? 1 : 0;
    
    tmin = (bounds[signs[0]].x - ray_origin.x) * ray_invdir.x;
    tmax = (bounds[1-signs[0]].x - ray_origin.x) * ray_invdir.x;
    tymin = (bounds[signs[1]].y - ray_origin.y) * ray_invdir.y;
    tymax = (bounds[1-signs[1]].y - ray_origin.y) * ray_invdir.y;
    
    if ((tmin > tymax) || (tymin > tmax))
        return HitInfo(false,0.0,0.0);

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    
    tzmin = (bounds[signs[2]].z - ray_origin.z) * ray_invdir.z;
    tzmax = (bounds[1-signs[2]].z - ray_origin.z) * ray_invdir.z;
    
    if ((tmin > tzmax) || (tzmin > tmax))
        return HitInfo(false,0.0,0.0);

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    return HitInfo(true,tmin,tmax);
}

float phase_fun(float g, float costheta)
{
    float g2 = g*g;
    return (1-g2) / pow(1+g2-2.0*g*costheta, 3/2) / (4.0 * PI);
}

float integrate_density_light(vec3 ray_origin, vec3 ray_dir, float step_size, float tmax)
{
    float tcurr = 0.0;
    float density = 0.0;
    vec3 curr = ray_origin;

    while(tcurr < tmax)
    {
        density += sigma_t(curr) * step_size;
        tcurr += step_size;
        curr += step_size * ray_dir;
    }

    return exp(-density);
}

float integrate_density(vec3 ray_origin, vec3 ray_dir, float step_size, float tmax)
{
    vec3 curr = ray_origin;
    float tcurr = 0.0;
    float density = 0.0;

    while(tcurr < tmax)
    {
        density += sigma_t(curr) * step_size; //sigma_t = 1.0
        tcurr += step_size;
        curr += step_size * ray_dir;

        //Get light contribution
        vec3 light_dir = light_position - curr;
        float light_dist = length(light_dir);
        light_dir /= light_dist;

        HitInfo intersection = intersect(curr, light_dir);

        float transmittance_light = integrate_density_light(curr, light_dir, step_size * 5, intersection.tmax);

        radiance += sigma_s(curr) * light_intensity / (light_dist * light_dist) * transmittance_light * phase_fun(g, dot(light_dir, ray_dir)); //Phase fun for isotropic scattering = 1/4pi
    }

    return exp(-density);
}

void main()
{
    vec3 ray_origin = frag_pos;
    vec3 ray_dir = normalize(frag_pos - camera_pos);

    HitInfo intersection = intersect(ray_origin + 1e-3*ray_dir, ray_dir);

    vec3 out_pos = ray_origin + intersection.tmax * ray_dir;

    //float transmittance_gt = exp(-sigma_t * intersection.tmax);
    float transmittance_march = integrate_density(ray_origin, ray_dir, 0.01, intersection.tmax);

    //radiance += transmittance_march * ambient;

    outColor = vec4(pow(vec3(1) - exp(-radiance), vec3(1.0/2.4)),1);
}