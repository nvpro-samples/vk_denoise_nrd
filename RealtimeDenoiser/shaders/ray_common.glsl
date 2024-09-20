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

#include "host_device.h"
#include "nvvkhl/shaders/dh_scn_desc.h"

// Useful for debugging results at individual pixels.
#define ATCURSOR(x)                                                                                                    \
  if(pixelPos == pc.mouseCoord)                                                                                        \
  {                                                                                                                    \
    x;                                                                                                                 \
  }

struct HitPayload
{
  uint  seed;
  float hitT;
  vec3  contrib;       // Output: Radiance (times MIS factors) at this point.
  vec3  weight;        // Output of closest-hit shader: BRDF sample weight of this bounce.
  vec3  rayOrigin;     // Input and output.
  vec3  rayDirection;  // Input and output.
  float bsdfPDF;       // Input and output: Probability that the BSDF sampling generated rayDirection.
};


struct HitPayloadNrd
{
  uint  renderNodeIndex;
  uint  renderPrimIndex;  // what mesh we hit
  float hitT;             // where we hit the mesh along the ray
  vec3  tangent;
  vec3  normal_envmapRadiance;  // when hitT == NRD_INF we hit the environment map and return its radiance here
  vec2  uv;
  float bitangentSign;
};


mat3 buildMirrorMatrix(vec3 normal)
{
  return mat3(-2.0 * (vec3(normal.x) * normal) + vec3(1.0, 0.0, 0.0),  //
              -2.0 * (vec3(normal.y) * normal) + vec3(0.0, 1.0, 0.0),  //
              -2.0 * (vec3(normal.z) * normal) + vec3(0.0, 0.0, 1.0));
}
