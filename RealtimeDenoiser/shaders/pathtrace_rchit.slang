/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "host_device.h"
#include "ray_common.slang"
#include "nrd_common.slang"
#include "get_hit.slang"
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/constants.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/pbr_material_types.h.slang"
#include "nvshaders/pbr_material_eval.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/ray_utils.h.slang"

// Bindings
[[vk::binding(RtxBindings::eTlas, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(SceneBindings::eTextures, 1)]] Sampler2D texturesMap[];
[[vk::binding(EnvBindings::eHdr, 3)]] Sampler2D hdrTexture;
[[vk::binding(EnvBindings::eImpSamples, 3)]] StructuredBuffer<EnvAccel> envSamplingData;

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pushConst;

struct ShadingResult
{
    float3 weight;
    float3 contrib;
    float3 rayOrigin;
    float3 rayDirection;
    float bsdfPDF;
};

// --------------------------------------------------------------------
// Sampling the Sun or the HDR
//
float3 sampleLights(in HitState state, inout uint seed, out float3 dirToLight, out float lightPdf)
{
    float3 randVal = float3(rand(seed), rand(seed), rand(seed));

    float3 lightContrib;

    if(TEST_FLAG(pushConst.frameInfo->flags, FLAGS_ENVMAP_SKY))
    {
        SkySamplingResult skySample;// = samplePhysicalSky(*pushConst->skyParams, randVal.xy);
        dirToLight = skySample.direction;
        lightPdf = skySample.pdf;
        lightContrib = skySample.radiance;
    }
    else
    {
        // Sample envmap in random direction, return direction in 'lightDir' and pdf in the sampled texture value
        float4 radiance_pdf = environmentSample(hdrTexture, envSamplingData, randVal, dirToLight);
        // rotate returned direction into worldspace
        dirToLight = rotate(dirToLight, float3(0, 1, 0), pushConst.frameInfo->envRotation);

        lightContrib = radiance_pdf.xyz;
        lightPdf = radiance_pdf.w;
    }

    lightContrib *= pushConst.frameInfo->envIntensity.xyz;

    return lightContrib / lightPdf;
}

//-----------------------------------------------------------------------
// Evaluate shading of 'pbrMat' at 'hit' position
//-----------------------------------------------------------------------
ShadingResult shading(in PbrMaterial pbrMat, in HitState hit, inout PayloadSecondary payload)
{
    ShadingResult result;

    // Emissive material contribution. No MIS here because we only use MIS for
    // skybox lighting.
    result.contrib = pbrMat.emissive;

    // Light contribution; can be environment or punctual lights
    float3 contribution = float3(0);
    float3 dirToLight = float3(0);
    float lightPdf = 0.0f;

    // Did we hit any light?
    float3 lightRadianceOverPdf = sampleLights(hit, payload.seed, dirToLight, lightPdf);

    // Is the light in front of the surface and has a valid contribution in the
    // chosen random direction?
    const bool lightValid = (dot(dirToLight, pbrMat.N) > 0.0f) && lightPdf > 0.0f;

    // Evaluate BSDF
    if(lightValid)
    {
        BsdfEvaluateData evalData;
        evalData.k1 = -WorldRayDirection();
        evalData.k2 = dirToLight;
        evalData.xi = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));

        // Evaluate the material's response in the light's direction
        bsdfEvaluate(evalData, pbrMat);

        if(evalData.pdf > 0.0)
        {
            // We might hit the envmap in two ways, once via 'sampleLights()' as direct light sample;
            // once indirectly by following the material's BSDF for the next ray segment.
            // Therefore, make sure we correctly apply "Multiple Importance Sampling" to both
            // sampling strategies, expressed in 'lightPdf' and 'evalData.pdf'.
            const float misWeight = powerHeuristic(lightPdf, evalData.pdf);

            // sample weight
            const float3 w = lightRadianceOverPdf * misWeight;
            contribution += w * (evalData.bsdf_diffuse + evalData.bsdf_glossy);

            // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
            payload.hitT = 0.0f;
            float3 shadowRayOrigin = offsetRay(hit.pos, hit.geonrm);

            RayDesc shadowRay;
            shadowRay.Origin = shadowRayOrigin;
            shadowRay.Direction = dirToLight;
            shadowRay.TMin = 0.001;
            shadowRay.TMax = NRD_INF;

            TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                     0xFF, SBTOFFSET_PATHTRACE, 0, MISSINDEX_PATHTRACE, shadowRay, payload);
                     
            // If hitting nothing, add light contribution
            if(abs(payload.hitT) == NRD_INF)
            {
                result.contrib += contribution;
            }
            // Restore original hit distance, so we don't accidentally stop the path tracing right here
            payload.hitT = RayTCurrent();
        }
    }

    // Sample BSDF to suggest a follow-up ray for more indirect lighting
    {
        BsdfSampleData sampleData;
        sampleData.k1 = -WorldRayDirection();  // to eye direction
        sampleData.xi = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
        bsdfSample(sampleData, pbrMat);

        if(sampleData.event_type == BSDF_EVENT_ABSORB)
        {
            // stop path, yet return the hit distance
            payload.hitT = -payload.hitT;
        }
        else
        {
            result.weight = sampleData.bsdf_over_pdf;
            result.rayDirection = sampleData.k2;
            result.bsdfPDF = sampleData.pdf;
            float3 offsetDir = dot(result.rayDirection, pbrMat.N) > 0 ? hit.geonrm : -hit.geonrm;
            result.rayOrigin = offsetRay(hit.pos, offsetDir);
        }
    }

    return result;
}

[shader("closesthit")]
void main(inout PayloadSecondary payload, in BuiltInTriangleIntersectionAttributes attr)
{
    uint instanceID = InstanceIndex();
    uint renderPrimID = InstanceID();
    uint triangleID = PrimitiveIndex();

    // Retrieve the Primitive mesh buffer information
    GltfRenderNode renderNode = pushConst.gltfScene->renderNodes[instanceID];
    GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

    HitState hit = GetHitState(renderPrim, pushConst.bitangentFlip, attr.barycentrics);

    // Scene materials
    uint matIndex = max(0, renderNode.materialID);  // material of primitive mesh

    // Material of the object and evaluated material (includes textures)
    GltfShadeMaterial mat = pushConst.gltfScene->materials[matIndex];
    GltfTextureInfo *texInfos = pushConst.gltfScene->textureInfos;
    PbrMaterial pbrMat = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv, texturesMap, texInfos);

    // Override material
    if(pushConst.overrideRoughness > 0)
    {
        pbrMat.roughness = float2(clamp(pushConst.overrideRoughness, 0.001, 1.0));
        pbrMat.roughness *= pbrMat.roughness;
    }
    if(pushConst.overrideMetallic > 0)
        pbrMat.metallic = pushConst.overrideMetallic;

    payload.hitT = RayTCurrent();
    ShadingResult result = shading(pbrMat, hit, payload);

    payload.weight = result.weight;        // material's throughput at hitposition
    payload.contrib = result.contrib;       // radiance coming from hitposition
    payload.rayOrigin = result.rayOrigin;     // next ray segment's origin
    payload.rayDirection = result.rayDirection;  // and direction
    payload.bsdfPDF = result.bsdfPDF;       // PDF value that corresponds with chosen direction
}
