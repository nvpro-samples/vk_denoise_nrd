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

// by Jan Eric Kyprianidis <www.kyprianidis.com>

#include "host_device.h"

// clang-format off
[[vk::binding(TaaBindings::eInImage, 0)]]  RWTexture2D<float4>   iImage0;
[[vk::binding(TaaBindings::eOutImage, 0)]] RWTexture2D<float4> oImage;

struct PushConstant {
  float alpha;
};

[[vk::push_constant]] ConstantBuffer<PushConstant> params;
// clang-format on

// From https://www.shadertoy.com/view/tsjfRh
float3 TAA(int2 fragCoord)
{
  // Get the neighborhood min / max from this frame's render
  float3 center = iImage0[fragCoord].rgb;
  float3 minColor = center;
  float3 maxColor = center;
  
  for(int iy = -1; iy <= 1; ++iy)
  {
    for(int ix = -1; ix <= 1; ++ix)
    {
      if(ix == 0 && iy == 0)
        continue;

      int2 offsetUV = fragCoord + int2(ix, iy);
      float3 color = iImage0[offsetUV].rgb;
      minColor = min(minColor, color);
      maxColor = max(maxColor, color);
    }
  }

  // Get last frame's pixel and clamp it to the neighborhood of this frame
  float3 old = oImage[fragCoord].rgb;
  old = clamp(old, minColor, maxColor);

  // Interpolate from the clamped old color to the new color.
  float3 pixelColor = lerp(old, center, params.alpha);
  return pixelColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
  int2 fragCoord = int2(threadIdx.xy);
  int2 imgSize;
  oImage.GetDimensions(imgSize.x, imgSize.y);
  
  if(fragCoord.x >= imgSize.x || fragCoord.y >= imgSize.y)
    return;

  float3 R = TAA(fragCoord);

  oImage[fragCoord] = float4(R, 0);
}

