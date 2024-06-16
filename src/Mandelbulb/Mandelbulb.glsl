#version 460 core

layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) uniform image2D img_output;

uniform vec3 camera_position;
uniform mat4 V, P, VP, invVP;

vec3 vecPow(vec3 v, float n)
{
    float r = length(v);

    if(r < 1e-5)
    {
        return vec3(0);
    }

    float phi = atan(v.y, v.x);
    float theta = acos(v.z / r);

    return pow(r, n) * vec3(sin(n * theta) * cos(n * phi),
                            sin(n * theta) * sin(n * phi),
                            cos(n * theta));
}

bool insideMandelbulb(vec3 query)
{
    vec3 v = vec3(0);
    vec3 c = query;

    int n = 8;

    for(int i = 0; i < 20; ++i)
    {
        v = vecPow(v, float(n)) + c;

        if(dot(v, v) > 16.)
        {
            return false;
        }
    }

    return true;
}

vec3 traceRayVolume(vec3 origin, vec3 direction)
{
    float step_size = 0.005;
    float max_distance = 5.0;

    float depth = 0.0;

    int points_inside = 0;

    float density = 0.0;
    float sigma_t = 2.0;
    while(depth < max_distance)
    {
        depth += step_size;

        vec3 query = origin + depth * direction;

        if(insideMandelbulb(query))
        {
            density += step_size * sigma_t;
        }
    }

    return vec3(1.0 - exp(-density));
}

float distanceMandelbulb(vec3 query)
{
    //return length(query) - 1.0;
    vec3 v = vec3(0);
    vec3 c = query;

    float d = 1.0;

    float n = 8.0;
    float hit = 0.0;

    for(int i = 0; i < 10; ++i)
    {
        float r = length(v);

        if(r > 2.)
        {
            hit = 0.5 * log(r) * r / d;
            break; 
        }

        v = vecPow(v, float(n)) + c;
        d = pow(r, n - 1.0) * (n - 1.0) * d + 1.0;
    }

    return hit;
}

vec3 computeNormal(vec3 p)
{
    const float h = 1e-6; // replace by an appropriate value
    const vec2 k = vec2(1,-1);
    return normalize( k.xyy*distanceMandelbulb( p + k.xyy*h ) + 
                      k.yyx*distanceMandelbulb( p + k.yyx*h ) + 
                      k.yxy*distanceMandelbulb( p + k.yxy*h ) + 
                      k.xxx*distanceMandelbulb( p + k.xxx*h ) );
}

vec3 traceRay(vec3 origin, vec3 direction)
{
    int num_steps = 128;

    vec3 query = origin;

    for(int i = 0; i < num_steps; ++i)
    {
        float dist = distanceMandelbulb(query);

        if(dist < 0.005)
        {
            vec3 normal = computeNormal(query);
            return normal * 0.5 + 0.5;
        }

        query += dist * direction;
    }

    return vec3(0);
}

void main()
{
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(img_output);

    vec3 color = vec3(0);
    // MSAA
    for(int i = 0; i < 2; ++i)
    {
        for(int j = 0; j < 2; ++j)
        {
            vec2 uv = (vec2(pixel_coords) + vec2(i,j) / 4.0) / vec2(image_size);

            vec4 ndc_pos = vec4(2.0 * uv - 1.0, 0.0, 1.0);
            vec4 world_pos = invVP * ndc_pos;

            vec3 plane_pos = world_pos.xyz / world_pos.w;

            vec3 ray_origin = camera_position;
            vec3 ray_dir = normalize(plane_pos - ray_origin);

            color += traceRay(ray_origin, ray_dir);
        }
    }

    color /= 4.0;

    imageStore(img_output, pixel_coords, vec4(color, 1));
}