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

 #ifndef NRD_SLANG_H
 #define NRD_SLANG_H 1

#include "nvshaders/slang_types.h"
#include "nvshaders/functions.h.slang"

#include "NRD.hlsli"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// NRD
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


#define USE_SANITIZATION true
#define STL_RF0_DIELECTRICS 0.04
//#define USE_SQRT_ROUGHNESS 1

// FIXME: these need to be in sync with the ReblurSettings::HitDistanceParameters
//    struct HitDistanceParameters
//    {
//        float A = 3.0f;     // constant value (m)
//        float B = 0.1f;     // viewZ based linear scale (m / units) (1 m - 10 cm, 10 m - 1 m, 100 m - 10 m)
//        float C = 10.0f;    // roughness based scale, "> 1" to get bigger hit distance for low roughness
//        float D = -25.0f;   // roughness based exponential scale, "< 0", absolute value should be big enough to collapse "exp2( D * roughness ^ 2 )" to "~0" for roughness = 1
//    };
const static float4 gDiffHitDistParams = float4(3.0f, 0.1f, 20.0f, -25.0f);
const static float4 gSpecHitDistParams = float4(3.0f, 0.1f, 20.0f, -25.0f);

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// STL.hlsli
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

const static float FLT_MIN = 1e-15;
float       PositiveRcp(float x)
{
  return 1.0 / (max(x, FLT_MIN));
}

void ConvertBaseColorMetalnessToAlbedoRf0(float3 baseColor, float metalness, out float3 albedo, out float3 Rf0)
{
  // TODO: ideally, STL_RF0_DIELECTRICS needs to be replaced with reflectance "STL_RF0_DIELECTRICS = 0.16 * reflectance * reflectance"
  // see https://google.github.io/filament/Filament.html#toc4.8
  // see https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf (page 13)
  albedo = baseColor * saturate(1.0 - metalness);
  Rf0    = lerp(float3(STL_RF0_DIELECTRICS), baseColor, metalness);
}

#endif
