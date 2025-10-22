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

[[vk::binding(EnvBindings::eHdr, 3)]] Sampler2D hdrTexture;

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pc;

// The main miss shader will be executed when the primary rays misses any geometry
// and just hit the background envmap. The resulting color values will be recorded
// into the "DirectLighting" buffer.
[shader("miss")]
void main(inout PayloadPrimary payload)
{
    float3 envColor;

    if(TEST_FLAG(pc.frameInfo.flags, FLAGS_ENVMAP_SKY))
    {
        envColor = evalPhysicalSky(*pc.skyParams, WorldRayDirection());
    }
    else
    {
        float3 dir = rotate(WorldRayDirection(), float3(0, 1, 0), -pc.frameInfo.envRotation);
        float2 uv = getSphericalUv(dir);
        envColor = hdrTexture.SampleLevel(uv, 0).rgb;
    }

    envColor *= pc.frameInfo.envIntensity.xyz;

    // No need to deal with the PDF here since the primary surface trace is
    // performed noise-free.
    payload.normal_envmapRadiance = envColor;
    payload.hitT = NRD_INF;  // Ending trace
}
