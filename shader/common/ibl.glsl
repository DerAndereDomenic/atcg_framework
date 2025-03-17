uniform samplerCube irradiance_map;
uniform samplerCube prefilter_map;
uniform sampler2D lut;

vec3 image_based_lighting(vec3 base_color, float metallic, float roughness, vec3 normal, vec3 view_dir)
{
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, base_color, metallic);
    float NdotV = max(dot(normal, view_dir), 0.0);
    vec3 kS = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD = 1.0 - kS;
    vec3 irradiance = texture(irradiance_map, normal).rgb;
    vec3 color_diffuse = mix(base_color, vec3(0), metallic);
    vec3 diffuse = irradiance * color_diffuse;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-view_dir, normal);
    vec3 prefilteredColor = textureLod(prefilter_map, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 lutbrdf = texture(lut, vec2(NdotV, roughness)).rg;
    vec3 specular = prefilteredColor * (kS * lutbrdf.x + lutbrdf.y);
    vec3 ambient = (kD * diffuse + specular);

    return ambient;
}