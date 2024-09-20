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

// clang-format off
layout(location = 1) rayPayloadInEXT HitPayloadNrd payloadNrd;
layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 3, binding = eHdr) uniform sampler2D hdrTexture;
// clang-format on

// The main miss shader will be executed when the primaary rays misses any geometry
// and just hit the background envmap. The resulting color values will be recorded
// into the "DirectLighting" buffer.
void main()
{
  vec3 dir = rotate(gl_WorldRayDirectionEXT, vec3(0, 1, 0), -frameInfo.envRotation);
  vec2 uv  = getSphericalUv(dir);
  vec3 env = texture(hdrTexture, uv).rgb;

  // No need to deal with the PDF here since the primary surface trace is
  // performed noise-free.
  payloadNrd.normal_envmapRadiance = env * frameInfo.clearColor.xyz;
  payloadNrd.hitT                  = NRD_INF;  // Ending trace
}
