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

#ifndef NRD_GLSL
#define NRD_GLSL 1  // This also instructs NRD.hlsli to use GLSL compatible lingo

#include "nvvkhl/shaders/func.h"

#include "NRDEncoding.hlsli"
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
vec4 gDiffHitDistParams = vec4(3.0f, 0.1f, 20.0f, -25.0f);
vec4 gSpecHitDistParams = vec4(3.0f, 0.1f, 20.0f, -25.0f);

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// STL.hlsli
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

const float FLT_MIN = 1e-15;
float       PositiveRcp(float x)
{
  return 1.0 / (max(x, FLT_MIN));
}

// "Ray Tracing Gems", Chapter 32, Equation 4 - the approximation assumes GGX VNDF and Schlick's approximation
float3 EnvironmentTerm_Rtg(float3 Rf0, float NoV, float linearRoughness)
{
  float m = saturate(linearRoughness * linearRoughness);

  vec4 X;
  X.x = 1.0;
  X.y = NoV;
  X.z = NoV * NoV;
  X.w = NoV * X.z;

  vec4 Y;
  Y.x = 1.0;
  Y.y = m;
  Y.z = m * m;
  Y.w = m * Y.z;

  mat2 M1 = mat2(0.99044, -1.28514, 1.29678, -0.755907);
  mat3 M2 = mat3(1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627);

  mat2 M3 = mat2(0.0365463, 3.32707, 9.0632, -9.04756);
  mat3 M4 = mat3(1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123);

  float bias  = dot(mul(M1, X.xy), Y.xy) * PositiveRcp(dot(mul(M2, X.xyw), Y.xyw));
  float scale = dot(mul(M3, X.xy), Y.xy) * PositiveRcp(dot(mul(M4, X.xzw), Y.xyw));

  return saturate(Rf0 * scale + bias);
}

#define STL_RF0_DIELECTRICS 0.04

void ConvertBaseColorMetalnessToAlbedoRf0(vec3 baseColor, float metalness, out vec3 albedo, out vec3 Rf0)
{
  // TODO: ideally, STL_RF0_DIELECTRICS needs to be replaced with reflectance "STL_RF0_DIELECTRICS = 0.16 * reflectance * reflectance"
  // see https://google.github.io/filament/Filament.html#toc4.8
  // see https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf (page 13)
  albedo = baseColor * saturate(1.0 - metalness);
  Rf0    = mix(vec3(STL_RF0_DIELECTRICS), baseColor, metalness);
}

#endif
