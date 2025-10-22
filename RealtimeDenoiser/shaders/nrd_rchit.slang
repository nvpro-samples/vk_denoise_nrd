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
#include "get_hit.slang"

[[vk::push_constant]] ConstantBuffer<RtxPushConstant> pc;

[shader("closesthit")]
void main(inout PayloadPrimary payload, in BuiltInTriangleIntersectionAttributes attr)
{
    uint instanceID = InstanceIndex();
    uint renderPrimID = InstanceID();
    uint triangleID = PrimitiveIndex();

    // Retrieve the Primitive mesh buffer information
    GltfRenderNode renderNode = pc.gltfScene->renderNodes[instanceID];
    GltfRenderPrimitive renderPrim = pc.gltfScene->renderPrimitives[renderPrimID];

    HitState hit = GetHitState(renderPrim, pc.bitangentFlip, attr.barycentrics);

    payload.renderNodeIndex = instanceID;
    payload.renderPrimIndex = renderPrimID;
    payload.tangent = hit.tangent;
    payload.bitangentSign = hit.bitangentSign;
    payload.hitT = RayTCurrent();
    payload.normal_envmapRadiance = hit.nrm;
    payload.uv = hit.uv;
}
