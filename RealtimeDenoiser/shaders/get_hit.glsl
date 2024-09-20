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
// This function returns the geometric information at hit point
// Note: depends on the buffer layout PrimMeshInfo

#ifndef GETHIT_GLSL
#define GETHIT_GLSL

#include "nvvkhl/shaders/vertex_accessor.h"
#include "nvvkhl/shaders/func.h"

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3  pos;
  vec3  nrm;
  vec3  geonrm;
  vec2  uv;
  vec3  tangent;
  vec3  bitangent;
  float bitangentSign;
};


//--------------------------------------------------------------
// Flipping Back-face
vec3 adjustShadingNormalToRayDir(inout vec3 N, inout vec3 G)
{
  const vec3 V = -gl_WorldRayDirectionEXT;

  if(dot(G, V) < 0)  // Flip if back facing
    G = -G;

  if(dot(G, N) < 0)  // Make Normal and GeoNormal on the same side
    N = -N;

  return N;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
HitState GetHitState(RenderPrimitive renderPrim)
{
  HitState hit;

  // Barycentric coordinate on the triangle
  vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = getTriangleIndices(renderPrim, gl_PrimitiveID);

  // Position
  const vec3 pos0     = getVertexPosition(renderPrim, triangleIndex.x);
  const vec3 pos1     = getVertexPosition(renderPrim, triangleIndex.y);
  const vec3 pos2     = getVertexPosition(renderPrim, triangleIndex.z);
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       worldGeoNormal = normalize(vec3(geoNormal * gl_WorldToObjectEXT));
  hit.geonrm                = worldGeoNormal;

  hit.nrm = worldGeoNormal;
  if(hasVertexNormal(renderPrim))
  {
    const vec3 normal      = getInterpolatedVertexNormal(renderPrim, triangleIndex, barycentrics);
    vec3       worldNormal = normalize(vec3(normal * gl_WorldToObjectEXT));
    adjustShadingNormalToRayDir(worldNormal, worldGeoNormal);
    hit.nrm = worldNormal;
  }

  // TexCoord
  hit.uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

  // Tangent - Bitangent
  vec4 tng[3];
  if(hasVertexTangent(renderPrim))
  {
    tng[0] = getVertexTangent(renderPrim, triangleIndex.x);
    tng[1] = getVertexTangent(renderPrim, triangleIndex.y);
    tng[2] = getVertexTangent(renderPrim, triangleIndex.z);
  }
  else
  {
    vec4 t = makeFastTangent(hit.nrm);
    tng[0] = t;
    tng[1] = t;
    tng[2] = t;
  }

  {
    hit.tangent   = normalize(mixBary(tng[0].xyz, tng[1].xyz, tng[2].xyz, barycentrics));
    hit.tangent   = vec3(gl_ObjectToWorldEXT * vec4(hit.tangent, 0.0));
    hit.tangent   = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
    hit.bitangent = cross(hit.nrm, hit.tangent) * tng[0].w;
    hit.bitangentSign = tng[0].w;
  }

  return hit;
}


#endif
