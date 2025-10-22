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
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/pbr_material_types.h.slang"
#include "nvshaders/pbr_material_eval.h.slang"
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/tonemap_functions.h.slang"

#include "nrd_common.slang"

// Individual binding points
[[vk::binding(RtxBindings::eTlas, 0)]] RaytracingAccelerationStructure topLevelAS;

[[vk::binding(SceneBindings::eTextures, 1)]] Sampler2D texturesMap[];

// Store direct lighting from emissive surfaces, lights and env. map
[[vk::binding(NrdBindings::eDirectLighting, 2)]] RWTexture2D<float4> nrdDirectLighting;
// Store object motion (currently not utilized)
[[vk::binding(NrdBindings::eMotionVectors, 2)]] RWTexture2D<float4> nrdObjectMotion;
// Store NRD's normal and roughness encoding
[[vk::binding(NrdBindings::eNormal_Roughness, 2)]] RWTexture2D<float4> nrdNormalRoughness;
// Store linear-Z along the camera axis in viewspace
[[vk::binding(NrdBindings::eViewZ, 2)]] RWTexture2D<float4> nrdViewZ;
// Store the noisy demodulated diffuse lighting
[[vk::binding(NrdBindings::eNoisyDiffuse, 2)]] RWTexture2D<float4> nrdUDiff;
// Store the noisy demodulated specular lighting
[[vk::binding(NrdBindings::eNoisySpecular, 2)]] RWTexture2D<float4> nrdUSpec;
// Store the material base color and metalness (to be used in composition)
[[vk::binding(NrdBindings::eBaseColor_Metalness, 2)]] RWTexture2D<float4> nrdBaseColorMetalness;

[[vk::binding(EnvBindings::eImpSamples, 3)]] StructuredBuffer<EnvAccel> envSamplingData;
[[vk::binding(EnvBindings::eHdr, 3)]] Sampler2D hdrTexture;

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pc;

struct HitState
{
    float3 pos;
    float3 nrm;
    float3 geonrm;
    float2 uv;
    float3 tangent;
    float3 bitangent;
    float bitangentSign;
};

//-----------------------------------------------------------------------
// Direct contribution from all lights (no HDR environment)
//-----------------------------------------------------------------------
float3 DirectLight(PbrMaterial matEval, HitState hitState, float3 toEye)
{
    float3 contribRadiance = float3(0);
#if NB_LIGHTS > 0

    uint nbLight = NB_LIGHTS;

    for(int light_index = 0; light_index < nbLight; light_index++)
    {
        Light light = pc.frameInfo->light[light_index];

        float3 lightDir;
        float3 lightContrib = lightContribution(light, hitState.pos, hitState.nrm, lightDir);
        float lightDist = (light.type != 0) ? 1e37f : length(hitState.pos - light.position);
        float dotNL = dot(lightDir, hitState.nrm);

        if(dotNL > 0.0)
        {
            float lightPdf = 1.0f / float(NB_LIGHTS);

            float pdf = 0;
            float3 brdf = pbrEval(matEval, toEye, lightDir, pdf);
            float3 radiance = brdf * dotNL * lightContrib / lightPdf;

            // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
            PayloadSecondary payload;
            payload.hitT = 0;

            RayDesc ray;
            ray.Origin = hitState.pos;
            ray.Direction = lightDir;
            ray.TMin = 0.001;
            ray.TMax = lightDist;

            TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                     0xFF, SBTOFFSET_PATHTRACE, 0, MISSINDEX_PATHTRACE, ray, payload);
                     
            // If hitting nothing, add light contribution
            if(abs(payload.hitT) == NRD_INF)
            {
                contribRadiance += radiance;
            }
        }
    }
#endif

    return contribRadiance;
}

//-----------------------------------------------------------------------
// Direct contribution of the HDR environment
//-----------------------------------------------------------------------
void HdrContrib(PbrMaterial pbrMat, float3 startPos, float3 toEye, out float3 diffuseRadiance, out float3 specularRadiance, inout PayloadSecondary payload)
{
    diffuseRadiance = float3(0);
    specularRadiance = float3(0);

    float3 lightDir;
    float3 lightContrib;
    float lightPdf;;

    float3 randVal = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
    
    if(TEST_FLAG(pc.frameInfo->flags, FLAGS_ENVMAP_SKY))
    {
        SkySamplingResult skySample = samplePhysicalSky(*pc.skyParams, randVal.xy);
        lightDir = skySample.direction;
        lightContrib = skySample.radiance;
        lightPdf = skySample.pdf;
    }
    else
    {
        // Sample envmap in random direction
        float4 radiance_pdf = environmentSample(hdrTexture, envSamplingData, randVal, lightDir);
        // rotate returned direction into worldspace
        lightDir = rotate(lightDir, float3(0, 1, 0), pc.frameInfo->envRotation);

        lightContrib = radiance_pdf.xyz;
        lightPdf = radiance_pdf.w;
    }

    lightContrib *= pc.frameInfo->envIntensity.xyz;

    float dotNL = dot(lightDir, pbrMat.N);

    // above surface?
    if(dotNL > 0.0 && lightPdf > 0.0)
    {
        float3 diffRadiance;
        float3 specRadiance;

        BsdfEvaluateData bsdfEval;
        bsdfEval.k1 = toEye;
        bsdfEval.k2 = lightDir;
        bsdfEval.xi = randVal;

        bsdfEvaluate(bsdfEval, pbrMat);

        if(bsdfEval.pdf > 0.0)
        {
            // We are potentially going to sample the environment map twice: once
            // via direct sampling (as performed here) using the envmap's PDF.
            // The other time is when following the pathtracer via the BSDF's PDF and
            // hitting the environment map.
            const float mis_weight = powerHeuristic(lightPdf, bsdfEval.pdf);

            float3 lightRadiance = mis_weight * lightContrib / lightPdf;
            // Material's diffuse response to envmap irradiance
            diffRadiance = bsdfEval.bsdf_diffuse * lightRadiance;
            // Material's specular/glossy response to envmap irradiance
            specRadiance = bsdfEval.bsdf_glossy * lightRadiance;

            // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
            payload.hitT = 0;
            
            RayDesc ray;
            ray.Origin = startPos;
            ray.Direction = lightDir;
            ray.TMin = 0.001;
            ray.TMax = NRD_INF;

            TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                     0xFF, SBTOFFSET_PATHTRACE, 0, MISSINDEX_PATHTRACE, ray, payload);

            // If ray to sky is not blocked, this is the environment light contribution
            // coming off the surface's location.
            if(abs(payload.hitT) == NRD_INF)
            {
                diffuseRadiance = diffRadiance;
                specularRadiance = specRadiance;
            }
        }
    }
}

//-----------------------------------------------------------------------
// Build Hit information from the payload's returned data and evaluate the
// material at the hit position
//-----------------------------------------------------------------------
void buildHitInfo(PayloadPrimary payloadPrimary, float3 rayOrigin, float3 rayDirection, out PbrMaterial pbrMat, out HitState hitState)
{
    // Retrieve the Primitive mesh buffer information
    GltfRenderNode renderNode = pc.gltfScene->renderNodes[payloadPrimary.renderNodeIndex];
    GltfRenderPrimitive renderPrim = pc.gltfScene->renderPrimitives[payloadPrimary.renderPrimIndex];

    // Calculate hitState position, normal tangent etc
    hitState.pos = rayOrigin + payloadPrimary.hitT * rayDirection;
    hitState.nrm = payloadPrimary.normal_envmapRadiance;
    hitState.geonrm = hitState.nrm;
    hitState.uv = payloadPrimary.uv;
    hitState.tangent = payloadPrimary.tangent;
    hitState.bitangent = cross(hitState.nrm, hitState.tangent) * payloadPrimary.bitangentSign;

    // Scene materials
    uint matIndex = max(0, renderNode.materialID);  // material of primitive mesh
    GltfShadeMaterial mat = pc.gltfScene->materials[matIndex];
    GltfTextureInfo *texInfos = pc.gltfScene->textureInfos;
    pbrMat = evaluateMaterial(mat, hitState.nrm, hitState.tangent, hitState.bitangent, hitState.uv, texturesMap, texInfos);

    if(pc.overrideRoughness > 0)
    {
        pbrMat.roughness = float2(clamp(pc.overrideRoughness, MICROFACET_MIN_ROUGHNESS, 1.0));
        pbrMat.roughness *= pbrMat.roughness;
    }
    if(pc.overrideMetallic > 0)
    {
        pbrMat.metallic = pc.overrideMetallic;
    }
}

//-----------------------------------------------------------------------
// ENTRY function
//-----------------------------------------------------------------------
[shader("raygeneration")]
void main()
{
    int2 pixelPos = int2(DispatchRaysIndex().xy);

    // Initialize the random number
    PayloadSecondary payload;
    payload.seed = xxhash32(uint3(DispatchRaysIndex().xy, pc.frame));

    float2 pixelCenter = float2(DispatchRaysIndex().xy) + 0.5;

    pixelCenter += pc.frameInfo->jitter;

    const float2 inUV = pixelCenter / float2(DispatchRaysDimensions().xy);
    const float2 d = inUV * 2.0 - 1.0;
    float3 origin = mul(float4(0.0, 0.0, 0.0, 1.0), pc.frameInfo->viewInv).xyz;
    const float3 eyePos = origin.xyz;
    const float4 target = mul(float4(d.x, d.y, 0.01, 1.0), pc.frameInfo->projInv);
    float3 direction = normalize(mul(target.xyz, (float3x3)pc.frameInfo->viewInv));
    const float3 orgDirection = direction;
    float3 toEye = -direction.xyz;
    const uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

    PbrMaterial pbrMat;  // Material at hitState position
    HitState hitState;

    // Result of trace
    bool hitSky = false;
    bool isPsr = false;
    float psrHitDist = 0.0;
    float3 psrThroughput = float3(1.0);
    float3 psrDirectRadiance = float3(0.0);
    float3x3 psrMirror = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);  // identity

    //====================================================================================================================
    // STEP 1 - Find first non-mirror primary hit.
    // The first non-mirror hit surface is used as 'Primary Surface Replacement'.
    // Collect G-Buffer material & hitState information.
    // #PSR
    //====================================================================================================================
    int psrDepth = 0;
    PayloadPrimary payloadPrimary;
    
    do
    {
        payloadPrimary.hitT = 0;
        payloadPrimary.renderNodeIndex = 0;
        payloadPrimary.renderPrimIndex = 0; 
        
        // Trace using nrd.rchit, nrd.rmiss and PayloadPrimary
        RayDesc ray;
        ray.Origin = origin.xyz;
        ray.Direction = direction.xyz;
        ray.TMin = 0.01;
        ray.TMax = 1e32;
        
        TraceRay(topLevelAS, rayFlags, 0xFF, SBTOFFSET_NRD, 0, MISSINDEX_NRD, ray, payloadPrimary);

        hitSky = (payloadPrimary.hitT == NRD_INF);
        if(hitSky)
        {
            psrDirectRadiance += psrThroughput * payloadPrimary.normal_envmapRadiance;
            break;
        }

        // Accumulate the hit distances along the mirrored reflections - used to calculate the
        // virtual world PSR position's ViewZ distance
        psrHitDist += payloadPrimary.hitT;

        buildHitInfo(payloadPrimary, origin, direction, pbrMat, hitState);
        origin = offsetRay(hitState.pos, pbrMat.Ng);

        // Did we hit anything other than a mirror?
        if((pbrMat.roughness.x > ((MICROFACET_MIN_ROUGHNESS * MICROFACET_MIN_ROUGHNESS) + 0.001)) || pbrMat.metallic < 1.0)
        {
            break;
        }

        // At least one mirror hit
        isPsr = true;

        // Only the glossy part should be non-zero as this is a mirror surface.
        // The pdf for the mirrored reflection should be infinity
        psrDirectRadiance += psrThroughput * pbrMat.emissive;

        {
            BsdfSampleData specBsdfSample;
            specBsdfSample.xi = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
            specBsdfSample.k1 = -direction;

            bsdfSample(specBsdfSample, pbrMat);

            psrThroughput *= specBsdfSample.bsdf_over_pdf;
            psrMirror = mul(psrMirror, buildMirrorMatrix(pbrMat.N));

            // Follow the mirror
            direction = reflect(direction, pbrMat.N);
        }

        ++psrDepth;
    } while(psrDepth < 5);

    // Early out when hitting sky (even via mirrors)
    if(hitSky)
    {
        nrdDirectLighting[pixelPos] = float4(psrDirectRadiance, 0.0);
        nrdUDiff[pixelPos] = float4(0);
        nrdUSpec[pixelPos] = float4(0);
        nrdNormalRoughness[pixelPos] = float4(0);
        nrdViewZ[pixelPos] = float4(-NRD_INF);
        return;
    }

    // ViewZ buffer
    float g_viewZ = mul(float4(eyePos + orgDirection * psrHitDist, 1.0), pc.frameInfo->view).z;  // NOTE: viewZ is the 'Z' of the world hitState position in camera space
    nrdViewZ[pixelPos] = float4(g_viewZ);

    // Normal/Roughness buffer
    {
        float materialType = (isPsr ? MATERIAL_ID_PSR : (pbrMat.metallic == 1.0 ? MATERIAL_ID_METAL : MATERIAL_ID_DEFAULT));

        // Transform surface normal from "virtual world normal" to world normal through a series of mirror-matrix.
        // In case of NOT hitting any mirror, 'psrMirror' is just the identity matrix
        float3 worldNormal = mul(pbrMat.N, psrMirror);

        float4 normalRoughness = NRD_FrontEnd_PackNormalAndRoughness(worldNormal, sqrt(pbrMat.roughness.x), materialType);
        nrdNormalRoughness[pixelPos] = normalRoughness;
    }

    // Tint the material by the accumulated tinting of the mirrors until we reached the PSR
    // 'psrThroughput' will be (1.0, 1.0, 1.0) when hitting no mirrors.
    pbrMat.baseColor *= psrThroughput;
    pbrMat.specularColor *= psrThroughput;
    pbrMat.emissive = pbrMat.emissive * psrThroughput + psrDirectRadiance;

    // Motion Vector Buffer
    nrdObjectMotion[pixelPos] = float4(0);

    // transform eye vector into "virtual world" for PSR surfaces (identity if primary hit is non-mirror material)
    // -direction happens to be the same direction as if we did 'toEye = toEye * psrMirror;'
    toEye = -direction;

    float VdotN = dot(toEye, pbrMat.N);
    float lobeWeights[LOBE_COUNT] = computeLobeWeights(pbrMat, VdotN, pbrMat.baseColor);
    float diffuseRatio = lobeWeights[LOBE_DIFFUSE_REFLECTION];
    float specularRatio = 1.0f - diffuseRatio;

    {
        // BaseColor/Metalness Buffer
        // Needed to reconstruct the full diffuse and specular color from demodulated radiance during composition
        float3 writeBaseColor = toSrgb(pbrMat.baseColor);
        nrdBaseColorMetalness[pixelPos] = float4(writeBaseColor, pbrMat.metallic);
    }

    //====================================================================================================================
    // STEP 2 - Get the direct light contribution at hitState position
    //====================================================================================================================

    // Getting contribution of HDR
    float3 hdrDiffuseRadiance = float3(0);
    float3 hdrSpecularRadiance = float3(0);

    HdrContrib(pbrMat, hitState.pos, toEye, hdrDiffuseRadiance, hdrSpecularRadiance, payload);

    // Contribution of all lights
    float3 directLum = DirectLight(pbrMat, hitState, toEye);

    directLum += psrDirectRadiance + pbrMat.emissive;

    nrdDirectLighting[pixelPos] = float4(directLum, 1);


    //====================================================================================================================
    // STEP 3 - Get the indirect diffuse contribution at hitState position
    // #DIFFUSE
    //====================================================================================================================
    {
        float3 diffuseAccum = hdrDiffuseRadiance;
        float pathLength = 0.0;  // if first hit creates absorbtion event, provide a hitdist of 0

        //====================================================================================================================
        // STEP 3.1 - Sampling direction for diffuse
        //====================================================================================================================

        BsdfSampleData diffBsdfSample;
        diffBsdfSample.xi = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
        diffBsdfSample.k1 = toEye;
        brdf_diffuse_sample(diffBsdfSample, pbrMat, pbrMat.baseColor);

        if(diffBsdfSample.event_type != BSDF_EVENT_ABSORB)
        {
            //====================================================================================================================
            // STEP 3.2 - Evaluation of throughput for the hitState out going direction
            //====================================================================================================================

            // Resetting payload
            payload.contrib = float3(0.0);
            payload.weight = float3(1.0);
            payload.hitT = NRD_INF;
            payload.rayDirection = diffBsdfSample.k2;
            payload.rayOrigin = origin;
            payload.bsdfPDF = diffBsdfSample.pdf;

            //====================================================================================================================
            // STEP 3.3 - Trace ray from depth 1 and path trace until the ray dies
            //====================================================================================================================
            float3 throughput = diffBsdfSample.bsdf_over_pdf * diffuseRatio;

            for(int depth = 1; depth < pc.maxDepth; depth++)
            {
                payload.hitT = NRD_INF;
                
                RayDesc secondaryRay;
                secondaryRay.Origin = payload.rayOrigin;
                secondaryRay.Direction = payload.rayDirection;
                secondaryRay.TMin = 0.001;
                secondaryRay.TMax = NRD_INF;
                
                TraceRay(topLevelAS, rayFlags, 0xFF, SBTOFFSET_PATHTRACE, 0, MISSINDEX_PATHTRACE, secondaryRay, payload);

                // Accumulating results
                diffuseAccum += payload.contrib * throughput;
                throughput *= payload.weight;

                // The first secondary path segment determines the hit distance.
                // If the ray hits the environment, NRD_INF is returned
                if(depth == 1)
                {
                    pathLength = abs(payload.hitT);
                }

                if(payload.hitT < 0.0)
                {
                    break;
                }
            }

            // Removing fireflies
            float lum = dot(diffuseAccum, float3(0.212671f, 0.715160f, 0.072169f));
            if(lum > pc.maxLuminance)
            {
                diffuseAccum *= pc.maxLuminance / lum;
            }
        }

        //====================================================================================================================
        // STEP 3.4 - Signal de-modulation
        //====================================================================================================================
        diffuseAccum /= (pbrMat.baseColor * 0.99 + 0.01);

        //====================================================================================================================
        // STEP 3.5 - Write accumulated
        //====================================================================================================================
        float4 diffIndirect = float4(0.0);

        if(pc.method == NRD_REBLUR)
        {
            pathLength = REBLUR_FrontEnd_GetNormHitDist(pathLength, g_viewZ, gDiffHitDistParams, 1.0);
            diffIndirect = REBLUR_FrontEnd_PackRadianceAndNormHitDist(diffuseAccum, pathLength, USE_SANITIZATION);
        }
        else if(pc.method == NRD_RELAX)
        {
            diffIndirect = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseAccum, pathLength, USE_SANITIZATION);
        }
        else
        {
            diffIndirect = float4(diffuseAccum, 1.0);
        }

        nrdUDiff[pixelPos] = diffIndirect;
    }

    //====================================================================================================================
    // STEP 4 - Get the indirect specular contribution at hitState position
    // #SPECULAR
    //====================================================================================================================

    {
        //====================================================================================================================
        // STEP 4.1 - Sampling direction for specular
        //====================================================================================================================
        float3 specularAccum = hdrSpecularRadiance;
        float pathLength = 0.0;  // if first hit creates absorbtion event, provide a hitdist of 0

        BsdfSampleData specBsdfSample;
        specBsdfSample.xi = float3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
        specBsdfSample.k1 = toEye;

        // HACK: Bias xi.z so that bsdfSample() only chooses between specular lobes.
        specBsdfSample.xi.z = (1.0f - lobeWeights[LOBE_DIFFUSE_REFLECTION]) * specBsdfSample.xi.z;
        bsdfSample(specBsdfSample, pbrMat);

        if(specBsdfSample.event_type != BSDF_EVENT_ABSORB)
        {
            //====================================================================================================================
            // STEP 4.2 - Evaluation of throughput for the hitState out going direction
            //====================================================================================================================

            // Resetting payload
            payload.contrib = float3(0.0);
            payload.weight = float3(1.0);
            payload.hitT = NRD_INF;
            payload.rayDirection = specBsdfSample.k2;
            payload.rayOrigin = origin;
            payload.bsdfPDF = specBsdfSample.pdf;

            //====================================================================================================================
            // STEP 4.3 - Trace ray from depth 1 and path trace until the ray dies
            //====================================================================================================================
            float3 throughput = specBsdfSample.bsdf_over_pdf * specularRatio;

            for(int depth = 1; depth < pc.maxDepth; depth++)
            {
                payload.hitT = -NRD_INF;
                
                RayDesc secondaryRay;
                secondaryRay.Origin = payload.rayOrigin;
                secondaryRay.Direction = payload.rayDirection;
                secondaryRay.TMin = 0.001;
                secondaryRay.TMax = NRD_INF;
                
                TraceRay(topLevelAS, rayFlags, 0xFF, SBTOFFSET_PATHTRACE, 0, MISSINDEX_PATHTRACE, secondaryRay, payload);

                // Accumulating results
                specularAccum += payload.contrib * throughput;
                throughput *= payload.weight;

                // The first secondary path segment determines the hit distance.
                // If the ray hits the environment, NRD_INF is returned
                if(depth == 1)
                {
                    pathLength = abs(payload.hitT);
                }

                // Breaking on end ray
                if(payload.hitT < 0.0)
                {
                    break;
                }
            }

            // Removing fireflies
            float lum = dot(specularAccum, float3(0.212671f, 0.715160f, 0.072169f));
            if(lum > pc.maxLuminance)
            {
                specularAccum *= pc.maxLuminance / lum;
            }
        }

        //====================================================================================================================
        // STEP 4.4 - Signal de-modulation
        //====================================================================================================================
        // Environment ( pre-integrated ) specular term
        float3 albedo, Rf0;
        ConvertBaseColorMetalnessToAlbedoRf0(pbrMat.baseColor, pbrMat.metallic, albedo, Rf0);
        float3 Fenv = _NRD_EnvironmentTerm_Rtg(Rf0, max(VdotN, 0.0), sqrt(pbrMat.roughness.x));
        specularAccum /= (Fenv * 0.99 + 0.01);

        //====================================================================================================================
        // STEP 4.5 - Write accumulated specular value to buffers
        //====================================================================================================================
        float4 specIndirect = float4(0.0);

        if(pc.method == NRD_REBLUR)
        {
            pathLength = REBLUR_FrontEnd_GetNormHitDist(pathLength, g_viewZ, gSpecHitDistParams, sqrt(pbrMat.roughness.x));
            specIndirect = REBLUR_FrontEnd_PackRadianceAndNormHitDist(specularAccum, pathLength, USE_SANITIZATION);
        }
        else if(pc.method == NRD_RELAX)
        {
            specIndirect = RELAX_FrontEnd_PackRadianceAndHitDist(specularAccum, pathLength, USE_SANITIZATION);
        }
        else
        {
            specIndirect = float4(specularAccum, 1.0);
        }

        // Store final specular color at pixel
        nrdUSpec[pixelPos] = specIndirect;
    }
}
