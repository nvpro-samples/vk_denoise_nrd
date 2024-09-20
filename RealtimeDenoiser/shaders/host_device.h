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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#define GRID_SIZE 16  // Grid size used by compute shaders

// clang-format off
#ifdef __cplusplus
  #include <glm/glm.hpp>
  using uint = uint32_t;

  using mat4 = glm::mat4;
  using vec4 = glm::vec4;
  using vec3 = glm::vec3;
  using vec2 = glm::vec2;
  using ivec2 = glm::ivec2;
  using uint = uint32_t;
#endif  // __cplusplus

#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

#define NB_LIGHTS 0

#define NRD_RELAX 0
#define NRD_REBLUR 1
#define NRD_REFERENCE 2

// We have two sets of shaders compiled into the Shader Binding Table;
// primary shaders, light-weight shaders used when finding the primary surface
// (which doesn't require random sampling), and pathtrace shaders, which are
// used for Monte Carlo path tracing.
#define PAYLOAD_NRD         1
#define PAYLOAD_PATHTRACE   0
#define SBTOFFSET_NRD       1
#define SBTOFFSET_PATHTRACE 0
#define MISSINDEX_NRD       1
#define MISSINDEX_PATHTRACE 0

START_BINDING(SceneBindings)
  eFrameInfo = 0,
  eSceneDesc = 1,
  eTextures  = 2
END_BINDING();

START_BINDING(RtxBindings)
  eTlas     = 0
END_BINDING();

START_BINDING(PostBindings)
  ePostImage       = 0
END_BINDING();

START_BINDING(NrdBindings)
  eViewZ                  = 0,
  eDirectLighting         = 1,
  eObjectMotion           = 2,
  eNormal_Roughness       = 3,
  eDiff                   = 4,
  eSpec                   = 5,
  eUnfiltered_Diff        = 6,
  eUnfiltered_Spec        = 7,
  eBaseColor_Metalness    = 8
END_BINDING();

START_BINDING(CompositionBindings)
  eCompImage = 0,
  eInDirect  = 1,
  eInDiff = 2,
  eInSpec = 3,
  eInBaseColor_Metalness = 4,
  eInNormal_Roughness = 5,
  eInViewZ = 6,
  eInFrameInfo = 7
END_BINDING();

START_BINDING(TaaBindings)
  eInImage = 0,
  eOutImage  = 1
END_BINDING();
// clang-format on

struct Light
{
  vec3  position;
  float intensity;
  vec3  color;
  int   type;
};

struct FrameInfo
{
  mat4  view;
  mat4  proj;
  mat4  viewInv;
  mat4  projInv;
  vec4  clearColor;
  vec2  jitter;
  float envRotation;
  float _pad;  // std430 layout requirements
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
  ivec2 mouseCoord;
};

#ifdef __cplusplus
#include <vulkan/vulkan_core.h>

inline VkExtent2D getGridSize(const VkExtent2D& size)
{
  return VkExtent2D{(size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE};
}
#endif

#endif  // HOST_DEVICE_H
