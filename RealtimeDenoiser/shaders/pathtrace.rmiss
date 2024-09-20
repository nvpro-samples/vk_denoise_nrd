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
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "nrd.glsl"

#include "host_device.h"
#include "ray_common.glsl"
#include "nvvkhl/shaders/func.h"
#include "nvvkhl/shaders/dh_hdr.h"

float powerHeuristic(float a, float b)
{
  const float t = a * a;
  return t / (b * b + t);
}


// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 3, binding = eHdr) uniform sampler2D hdrTexture;
// clang-format on

// If the pathtracer misses, it means the ray segment hit the environment map.
void main()
{
  vec3 dir = rotate(gl_WorldRayDirectionEXT, vec3(0, 1, 0), -frameInfo.envRotation);
  vec2 uv  = getSphericalUv(dir);
  vec4 env = texture(hdrTexture, uv);

  // From any surface point its possible to hit the environment map via two ways
  // a) as result from direct sampling or b) as result of following the material
  // BSDF. Here we deal with b). Calculate the proper MIS weight by taking the
  // BSDF's PDF in ray direction and the envmap's PDF in ray direction into account.
  float mis_weight = powerHeuristic(payload.bsdfPDF, env.a);
  payload.contrib  = mis_weight * env.rgb * frameInfo.clearColor.xyz;
  payload.hitT     = NRD_INF;  // Ending trace
}
