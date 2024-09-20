/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "host_device.h"
#include "ray_common.glsl"
#include "nvvkhl/shaders/dh_scn_desc.h"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 1) rayPayloadInEXT HitPayloadNrd payloadNrd;

layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };

layout(set = 0, binding = eTlas ) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures

layout(push_constant, scalar) uniform RtxPushConstant_  { RtxPushConstant pc; };
  // clang-format on

#include "nvvkhl/shaders/pbr_mat_struct.h"
#include "nvvkhl/shaders/pbr_mat_eval.h"
#include "get_hit.glsl"


// The main hit shader does not do anything other than report the hit back to the ray generation shader.
void main()
{
  // Retrieve the Primitive mesh buffer information
  RenderPrimitive renderPrim  = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[gl_InstanceCustomIndexEXT];

  HitState hit = GetHitState(renderPrim);

  payloadNrd.renderNodeIndex       = gl_InstanceID;
  payloadNrd.renderPrimIndex       = gl_InstanceCustomIndexEXT;
  payloadNrd.tangent               = hit.tangent;
  payloadNrd.bitangentSign         = hit.bitangentSign;
  payloadNrd.hitT                  = gl_HitTEXT;
  payloadNrd.normal_envmapRadiance = hit.nrm;
  payloadNrd.uv                    = hit.uv;
}
