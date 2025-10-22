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
#include "nvshaders/gltf_scene_io.h.slang"

// Useful for debugging results at individual pixels.
#define ATCURSOR(x)                                                                                                    \
  if(pixelPos == pc.mouseCoord)                                                                                        \
  {                                                                                                                    \
    x;                                                                                                                 \
  }

struct PayloadSecondary
{
  uint   seed;
  float  hitT;
  float3 contrib;       // Output: Radiance (times MIS factors) at this point.
  float3 weight;        // Output of closest-hit shader: BRDF sample weight of this bounce.
  float3 rayOrigin;     // Input and output.
  float3 rayDirection;  // Input and output.
  float  bsdfPDF;       // Input and output: Probability that the BSDF sampling generated rayDirection.
};


struct PayloadPrimary
{
  uint   renderNodeIndex;
  uint   renderPrimIndex;  // what mesh we hit
  float  hitT;             // where we hit the mesh along the ray
  float3 tangent;
  float3 normal_envmapRadiance;  // when hitT == NRD_INF we hit the environment map and return its radiance here
  float2 uv;
  float  bitangentSign;
};


float3x3 buildMirrorMatrix(float3 normal)
{
  return float3x3(-2.0 * (float3(normal.x) * normal) + float3(1.0, 0.0, 0.0),  //
                  -2.0 * (float3(normal.y) * normal) + float3(0.0, 1.0, 0.0),  //
                  -2.0 * (float3(normal.z) * normal) + float3(0.0, 0.0, 1.0));
}
