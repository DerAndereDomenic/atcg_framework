#version 330 core

#define INFINITY 1e20
#define PI 3.141596

layout (location = 0) out vec4 outColor;

in vec3 frag_pos_model;
flat in vec3 camera_pos_model;

float max_travel_distance(vec3 pos, vec3 dir)
{
    vec3 invDir = 1. / dir;
    int sig[3];
    vec3 bounds[2];
    bounds[0] = vec3(-1,-1,-1);
    bounds[1] = vec3(1,1,1);
    if(invDir.x < 0)
        sig[0] = 1;
    else 
        sig[0] = 0;

    if(invDir.y < 0)
        sig[1] = 1;
    else 
        sig[1] = 0;

    if(invDir.z < 0)
        sig[2] = 1;
    else 
        sig[2] = 0;

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (bounds[sig[0]].x - pos.x) * invDir.x; 
    tmax = (bounds[1-sig[0]].x - pos.x) * invDir.x; 
    tymin = (bounds[sig[1]].y - pos.y) * invDir.y; 
    tymax = (bounds[1-sig[1]].y - pos.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax)) 
        return INFINITY; 
 
    if (tymin > tmin) 
        tmin = tymin; 
    if (tymax < tmax) 
        tmax = tymax; 
 
    tzmin = (bounds[sig[2]].z - pos.z) * invDir.z; 
    tzmax = (bounds[1-sig[2]].z - pos.z) * invDir.z; 
 
    if ((tmin > tzmax) || (tzmin > tmax)) 
        return INFINITY; 
 
    if (tzmin > tmin) 
        tmin = tzmin; 
    if (tzmax < tmax) 
        tmax = tzmax; 
 
    return tmax;
}

float phase_fun(float cos_theta, float g)
{
    if (abs(g) < 1e-3)
        return 0.25 / PI;

    float g2 = g * g;
    float area = 4.0 * PI;
    return (1. - g2) / area * pow((1. + g2 - 2.0 * g * cos_theta), -1.5);
}

void main()
{
    vec3 ray_pos = frag_pos_model;
    vec3 ray_dir = normalize(frag_pos_model - camera_pos_model);

    vec3 light_pos = vec3(2,0,2);

    float dist = max_travel_distance(ray_pos, ray_dir);

    if(dist == INFINITY)
    {
        outColor = vec4(vec3(76., 76., 128.) / 255., 1);
        return;
    }

    float sigma_a = 0.;
    float sigma_s = 1.;
    float density = sigma_a + sigma_s; //sigma_t

    float travel_distance = 0.;
    float step_size = 0.01;
    float collected_density = 0.;

    vec3 Li = vec3(0);//vec3(76., 76., 128.) / 255.;
    float g = 0.;

    float light_intensity = 1.;

    while(travel_distance < dist)
    {
        travel_distance += step_size;
        collected_density += density * step_size;

        vec3 sample_pos = ray_pos + travel_distance * ray_dir;
        vec3 light_dir = normalize(light_pos - sample_pos);
        float light_r = length(light_pos - sample_pos);
        float Le = light_intensity / (light_r * light_r);

        float light_dist = max_travel_distance(sample_pos, light_dir);

        float cos_theta = dot(ray_dir, light_dir);

        float Ls = Le * phase_fun(cos_theta, g) * sigma_s * exp(-light_dist * density);

        Li += exp(-step_size * density) * Ls;

        //Li += Ls*;
    }

    float T = exp(-dist * density);
    vec3 Lo = Li + T * vec3(76., 76., 128.) / 255.;

    outColor = vec4(Lo, 1);
}