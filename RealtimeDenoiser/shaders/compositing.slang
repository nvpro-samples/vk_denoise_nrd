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
#include "nrd_common.slang"
#include "nvshaders/tonemap_functions.h.slang"

// clang-format off
[[vk::binding(CompositionBindings::eCompImage, 0)]]             RWTexture2D<float4> oImage;
[[vk::binding(CompositionBindings::eInDirect, 0)]]              RWTexture2D<float4> iDirect;
[[vk::binding(CompositionBindings::eInDiffuse, 0)]]                RWTexture2D<float4> iDiff;
[[vk::binding(CompositionBindings::eInSpecular, 0)]]                RWTexture2D<float4> iSpec;
[[vk::binding(CompositionBindings::eInBaseColor_Metalness, 0)]] RWTexture2D<float4> iBaseColor_Metalness;
[[vk::binding(CompositionBindings::eInNormal_Roughness, 0)]]    RWTexture2D<float4> iNormal_Roughness;
[[vk::binding(CompositionBindings::eInViewZ, 0)]]               RWTexture2D<float4> iViewZ;

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pc;
// clang-format on

float3 ReconstructViewPosition(float2 uv, float viewZ)
{
  float4 p;
  p = mul(float4((uv * 2.0 - float2(1.0)), 0.0, 1.0), pc.frameInfo->projInv);
  p /= p.w;
  p.xyz *= viewZ / p.z;

  p = mul(p, pc.frameInfo->viewInv);

  return p.xyz;
}

// Composite final image from denoised diffuse and specular channels
// as well as the direct lighting channel
[shader("compute")]
[numthreads(GRID_SIZE, GRID_SIZE, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  int2 fragCoord = int2(threadIdx.xy);
  int2 imgSize;
  oImage.GetDimensions(imgSize.x, imgSize.y);
  
  if(fragCoord.x >= imgSize.x || fragCoord.y >= imgSize.y)
    return;

  float4 directLighting = iDirect[fragCoord];
  float3 R = float3(directLighting.rgb);

  if(directLighting.a > 0)  // directLight.a == 0 denotes "just skybox visible"
  {
    float3 indirectDiff;
    float3 indirectSpec;

    if(pc.method == NRD_REBLUR)
    {
      indirectDiff = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(iDiff[fragCoord]).rgb;
      indirectSpec = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(iSpec[fragCoord]).rgb;
    }
    else if(pc.method == NRD_RELAX)
    {
      indirectDiff = RELAX_BackEnd_UnpackRadiance(iDiff[fragCoord]).rgb;
      indirectSpec = RELAX_BackEnd_UnpackRadiance(iSpec[fragCoord]).rgb;
    }
    else  // Reference Denoiser
    {
      indirectDiff = iDiff[fragCoord].rgb;
      indirectSpec = iSpec[fragCoord].rgb;
    }

    // normalized pixel coordinate
    float2 pixelUv = (float2(fragCoord) + float2(0.5) + pc.frameInfo->jitter) / float2(imgSize);

    // Reconstruct pixel's world position
    float viewZ = iViewZ[fragCoord].x;
    float3 Pw = ReconstructViewPosition(pixelUv, viewZ);

    // view vector, normal vector, material roughness
    // This could likely be done simpler. We should not need Pw. V could be
    // derived from just fragCoord and the inverse projection matrix.
    float3 V = normalize(pc.frameInfo->viewInv[3].xyz - Pw);
    float4 N_roughness = NRD_FrontEnd_UnpackNormalAndRoughness(iNormal_Roughness[fragCoord]);

    // Material properties at pixel coordinate needed to do re-modulation of diffuse and specular
    float4 baseColorMetalness = iBaseColor_Metalness[fragCoord];
    float3 baseColor = toLinear(baseColorMetalness.rgb);

    float3 albedo, Rf0;
    ConvertBaseColorMetalnessToAlbedoRf0(baseColor, baseColorMetalness.w, albedo, Rf0);

    // Environment ( pre-integrated ) specular term
    float NoV = dot(N_roughness.xyz, V);
    float3 Fenv = _NRD_EnvironmentTerm_Rtg(Rf0, NoV, N_roughness.w);

    float3 diffDemodulate = baseColor * 0.99 + 0.01;
    float3 specDemodulate = Fenv * 0.99 + 0.01;

    // Composition
    R += indirectDiff * diffDemodulate;
    R += indirectSpec * specDemodulate;
  }

  oImage[fragCoord] = float4(R, 1.0);
}
