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
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/functions.h.slang"

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pc;
[[vk::binding(EnvBindings::eHdr, 3)]] Sampler2D hdrTexture;

// If the pathtracer misses, it means the ray segment hit the environment map.
[shader("miss")]
void main(inout PayloadSecondary payload)
{
    float3 envColor;
    float envPdf;

    if(TEST_FLAG(pc.frameInfo.flags, FLAGS_ENVMAP_SKY))
    {
        envColor = evalPhysicalSky(*pc->skyParams, WorldRayDirection());
        envPdf = samplePhysicalSkyPDF(*pc->skyParams, WorldRayDirection());
    }
    else
    {
        float3 dir = rotate(WorldRayDirection(), float3(0, 1, 0), -pc.frameInfo.envRotation);
        float2 uv = getSphericalUv(dir);
        float4 hdrColorPdf = hdrTexture.SampleLevel(uv, 0);
        envColor = hdrColorPdf.rgb;
        envPdf = hdrColorPdf.w;
    }

    envColor *= pc.frameInfo.envIntensity.xyz;

    // From any surface point its possible to hit the environment map via two ways
    // a) as result from direct sampling or b) as result of following the material
    // BSDF. Here we deal with b). Calculate the proper MIS weight by taking the
    // BSDF's PDF in ray direction and the envmap's PDF in ray direction into account.
    float mis_weight = powerHeuristic(payload.bsdfPDF, envPdf);
    payload.contrib = mis_weight * envColor;
    payload.hitT = -NRD_INF;  // Ending trace
}
