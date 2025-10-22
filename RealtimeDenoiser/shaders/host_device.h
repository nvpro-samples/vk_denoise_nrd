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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#define GRID_SIZE 16  // Grid size used by compute shaders

#include "nvshaders/slang_types.h"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/sky_io.h.slang"

// clang-format off
#ifdef __cplusplus
#include <vulkan/vulkan_core.h>
#endif  // __cplusplus

NAMESPACE_SHADERIO_BEGIN()

#define START_BINDING(a) enum a {
#define END_BINDING() }

#define NB_LIGHTS 0

#define NRD_RELAX 0
#define NRD_REBLUR 1
#define NRD_REFERENCE 2

// We have two sets of shaders compiled into the Shader Binding Table;
// primary shaders, light-weight shaders used when finding the primary surface
// (which doesn't require random sampling), and pathtrace shaders, which are
// used for Monte Carlo path tracing.
#define SBTOFFSET_NRD       0
#define SBTOFFSET_PATHTRACE 1
#define MISSINDEX_NRD       0
#define MISSINDEX_PATHTRACE 1

// NRD Material ID
// Think of these as 'material types' to help NRD differentiate different types of materials
#define MATERIAL_ID_DEFAULT 0
#define MATERIAL_ID_METAL 1
#define MATERIAL_ID_PSR 2
#define MATERIAL_ID_HAIR 3

START_BINDING(SceneBindings)
  eTextures
END_BINDING();

START_BINDING(RtxBindings)
  eTlas
END_BINDING();

START_BINDING(PostBindings)
  ePostImage       = 0
END_BINDING();

START_BINDING(NrdBindings)
  eViewZ                  ,
  eDirectLighting         ,
  eMotionVectors          ,
  eNormal_Roughness       ,
  eNoisyDiffuse        ,
  eNoisySpecular        ,
  eBaseColor_Metalness
END_BINDING();

START_BINDING(CompositionBindings)
  eCompImage ,
  eInDirect  ,
  eInDiffuse ,
  eInSpecular ,
  eInBaseColor_Metalness ,
  eInNormal_Roughness ,
  eInViewZ ,
END_BINDING();

START_BINDING(TaaBindings)
  eInImage ,
  eOutImage
END_BINDING();
// clang-format on

struct Light
{
  float3 position;
  float  intensity;
  float3 color;
  int    type;
};

#define TEST_FLAG(flags, flag) bool((flags) & (flag))

#define BIT(x) (1 << x)

#define FLAGS_ENVMAP_SKY BIT(0)
#define FLAGS_USE_PSR BIT(1)
#define FLAGS_USE_PATH_REGULARIZATION BIT(2)

struct FrameInfo
{
  float4x4 view;
  float4x4 proj;
  float4x4 viewInv;
  float4x4 projInv;
  float4x4 prevMVP;
  float4   envIntensity;
  float2   jitter;
  float    envRotation;
  uint     flags;
#if NB_LIGHTS > 0
  Light light[NB_LIGHTS];
#endif
};

struct RtxPushConstant
{
  int   frame;
  float maxLuminance;
  uint  maxDepth;
  int   method;
  float meterToUnitsMultiplier;
  float overrideRoughness;
  float overrideMetallic;
  int2  mouseCoord;
  float bitangentFlip;

  FrameInfo*             frameInfo;
  SkyPhysicalParameters* skyParams;
  GltfScene*             gltfScene;
};

#ifdef __cplusplus

inline VkExtent2D getGridSize(const VkExtent2D& size)
{
  return VkExtent2D{(size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE};
}

NAMESPACE_SHADERIO_END()

#endif

#endif  // HOST_DEVICE_H
