#version 330 core

#include "common/defines.glsl"

out vec4 FragColor;

in vec3 localPos;

uniform samplerCube skybox;

void main()
{
	vec3 normal = normalize(localPos);
	vec3 irradiance = vec3(0.0);

	vec3 up = vec3(0, 1, 0);
	vec3 right = normalize(cross(up, normal));
	up = normalize(cross(normal, right));

	float sampleDelta = 0.025;
	float nrSamples = 0.0;

	for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
	{
		for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
		{
			vec3 tangentSample = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
			vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal;

			irradiance += texture(skybox, sampleVec).rgb * cos(theta) * sin(theta);
			++nrSamples;
		}
	}

	irradiance = PI * irradiance * (1.0 / float(nrSamples));

	FragColor = vec4(irradiance, 1.0);
}