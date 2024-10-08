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

#include "nrd.glsl"

#include "host_device.h"
#include "ray_common.glsl"

#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/dh_hdr.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/random.h"
#include "nvvkhl/shaders/bsdf_functions.h"
#include "nvvkhl/shaders/ray_util.h"
#include "nvvkhl/shaders/vertex_accessor.h"


hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };

#include "get_hit.glsl"

layout(set = 0, binding = eTlas ) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures

layout(set = 3, binding = eImpSamples,  scalar)	buffer _EnvAccel { EnvAccel envSamplingData[]; };
layout(set = 3, binding = eHdr) uniform sampler2D hdrTexture;


layout(push_constant, scalar) uniform RtxPushConstant_  { RtxPushConstant pc; };
// clang-format on


#include "nvvkhl/shaders/pbr_mat_eval.h"  // texturesMap
#include "nvvkhl/shaders/hdr_env_sampling.h"

struct ShadingResult
{
  vec3  weight;
  vec3  contrib;
  vec3  rayOrigin;
  vec3  rayDirection;
  float bsdfPDF;
};

// --------------------------------------------------------------------
// Sampling the Sun or the HDR
//
vec3 sampleLights(in HitState state, inout uint seed, out vec3 dirToLight, out float lightPdf)
{
  vec3 rand_val     = vec3(rand(seed), rand(seed), rand(seed));
  vec4 radiance_pdf = environmentSample(hdrTexture, rand_val, dirToLight);
  vec3 radiance     = radiance_pdf.xyz;
  lightPdf          = radiance_pdf.w;

  // Apply rotation and environment intensity
  dirToLight = rotate(dirToLight, vec3(0, 1, 0), frameInfo.envRotation);
  radiance *= frameInfo.clearColor.xyz;

  return radiance / lightPdf;
}


//-----------------------------------------------------------------------
// Evaluate shading of  'pbrMat' at 'hit' position
//-----------------------------------------------------------------------
ShadingResult shading(in PbrMaterial pbrMat, in HitState hit)
{
  ShadingResult result;

  // Emissive material contribution. No MIS here because we only use MIS for
  // skybox lighting.
  result.contrib = pbrMat.emissive;

  // Light contribution; can be environment or punctual lights
  vec3  contribution = vec3(0);
  vec3  dirToLight   = vec3(0);
  float lightPdf     = 0.F;

  // Did we hit any light?
  vec3 lightRadianceOverPdf = sampleLights(hit, payload.seed, dirToLight, lightPdf);

  // Is the light in front of the surface and has a valid contribution in the
  // chosen random direction?
  const bool lightValid = (dot(dirToLight, pbrMat.N) > 0.0f) && lightPdf > 0.0f;

  // Evaluate BSDF
  if(lightValid)
  {
    BsdfEvaluateData evalData;
    evalData.k1 = -gl_WorldRayDirectionEXT;
    evalData.k2 = dirToLight;
    evalData.xi = vec3(rand(payload.seed), rand(payload.seed), rand(payload.seed));

    // Evaluate the material's response in the light's direction
    bsdfEvaluate(evalData, pbrMat);

    if(evalData.pdf > 0.0)
    {
      // We might hit the envmap in two ways, once via 'sampleLights()' as direct light sample;
      // once indirectly by following the material's BSDF for the next ray segment.
      // Therefore, make sure we correctly apply "Multiple Importance Sampling" to both
      // sampling strategies, expressed in 'lightPdf' and 'evalData.pdf'.
      const float misWeight = powerHeuristic(lightPdf, evalData.pdf);

      // sample weight
      const vec3 w = lightRadianceOverPdf * misWeight;
      contribution += w * (evalData.bsdf_diffuse + evalData.bsdf_glossy);

      // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
      uint ray_flag = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
      payload.hitT         = 0.0F;
      vec3 shadowRayOrigin = offsetRay(hit.pos, hit.geonrm);

      traceRayEXT(topLevelAS, ray_flag, 0xFF, SBTOFFSET_PATHTRACE, 0, MISSINDEX_PATHTRACE, shadowRayOrigin, 0.001,
                  dirToLight, NRD_INF, PAYLOAD_PATHTRACE);
      // If hitting nothing, add light contribution
      if(abs(payload.hitT) == NRD_INF)
      {
        result.contrib += contribution;
      }
      // Restore original hit distance, so we don't accidentally stop the path tracing right here
      payload.hitT = gl_HitTEXT;
    }
  }

  // Sample BSDF to suggest a follow-up ray for more indirect lighting
  {
    BsdfSampleData sampleData;
    sampleData.k1 = -gl_WorldRayDirectionEXT;  // to eye direction
    sampleData.xi = vec3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
    bsdfSample(sampleData, pbrMat);

    if(sampleData.event_type == BSDF_EVENT_ABSORB)
    {
      // stop path, yet return the hit distance
      payload.hitT = -payload.hitT;
    }
    else
    {
      result.weight       = sampleData.bsdf_over_pdf;
      result.rayDirection = sampleData.k2;
      result.bsdfPDF      = sampleData.pdf;
      vec3 offsetDir      = dot(result.rayDirection,  pbrMat.N) > 0 ? hit.geonrm : -hit.geonrm;
      result.rayOrigin    = offsetRay(hit.pos, offsetDir);
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  RenderNode      renderNode = RenderNodeBuf(sceneDesc.renderNodeAddress)._[gl_InstanceID];
  RenderPrimitive renderPrim = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[gl_InstanceCustomIndexEXT];

  HitState hit = GetHitState(renderPrim);

  // Scene materials
  uint      matIndex  = max(0, renderNode.materialID);  // material of primitive mesh
  Materials materials = Materials(sceneDesc.materialAddress);

  // Material of the object and evaluated material (includes textures)
  GltfShadeMaterial mat    = materials.m[matIndex];
  PbrMaterial       pbrMat = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  // Override material
  if(pc.overrideRoughness > 0)
  {
    pbrMat.roughness = vec2(clamp(pc.overrideRoughness, 0.001, 1.0));
    pbrMat.roughness *= pbrMat.roughness;
  }
  if(pc.overrideMetallic > 0)
    pbrMat.metallic = pc.overrideMetallic;

  payload.hitT         = gl_HitTEXT;
  ShadingResult result = shading(pbrMat, hit);

  payload.weight       = result.weight;        // material's throughput at hitposition
  payload.contrib      = result.contrib;       // radiance coming from hitposition
  payload.rayOrigin    = result.rayOrigin;     // next ray segment's origin
  payload.rayDirection = result.rayDirection;  // and direction
  payload.bsdfPDF      = result.bsdfPDF;       // PDF value that corresponds with chosen direction
}
