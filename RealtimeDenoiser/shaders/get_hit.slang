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

#ifndef GETHIT_SLANG
#define GETHIT_SLANG

#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/gltf_vertex_access.h.slang"
#include "nvshaders/functions.h.slang"

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  float3 pos;
  float3 nrm;
  float3 geonrm;
  float2 uv;
  float3 tangent;
  float3 bitangent;
  float  bitangentSign;
};


//--------------------------------------------------------------
// Flipping Back-face
float3 adjustShadingNormalToRayDir(inout float3 N, inout float3 G)
{
  const float3 V = -WorldRayDirection();

  if(dot(G, V) < 0)  // Flip if back facing
    G = -G;

  if(dot(G, N) < 0)  // Make Normal and GeoNormal on the same side
    N = -N;

  return N;
}

float3x2 getTexCoords0(GltfRenderPrimitive renderPrim, uint3 idx)
{
  if(!hasVertexTexCoord0(renderPrim))
    return float3x2(0);

  float2*  texcoords = renderPrim.vertexBuffer.texCoords0;
  float3x2 uv;
  uv[0] = texcoords[idx.x];
  uv[1] = texcoords[idx.y];
  uv[2] = texcoords[idx.z];
  return uv;
}

float2 getInterpolatedVertexTexCoords(GltfRenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if(!hasVertexTexCoord0(renderPrim))
    return float2(0, 0);

  float3x2 uv = getTexCoords0(renderPrim, idx);

  return uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;
}


void computeTangentSpace(GltfRenderPrimitive renderPrim, uint3 idx, inout HitState hit)
{
  float3x2 uv = getTexCoords0(renderPrim, idx);

  float2 u = uv[1] - uv[0];
  float2 v = uv[2] - uv[0];

  float d = u.x * v.y - u.y * v.x;
  if(d == 0.0f)
  {
    float4 t          = makeFastTangent(hit.nrm);
    hit.tangent       = t.xyz;
    hit.bitangent     = cross(hit.nrm, hit.tangent) * t.w;
    hit.bitangentSign = t.w;
  }
  else
  {
    u /= d;
    v /= d;

    float3 v0 = getVertexPosition(renderPrim, idx.x);
    float3 v1 = getVertexPosition(renderPrim, idx.y);
    float3 v2 = getVertexPosition(renderPrim, idx.z);

    float3 p = v1 - v0;
    float3 q = v2 - v0;

    float3 t;
    t.x = v.y * p.x - u.y * q.x;
    t.y = v.y * p.y - u.y * q.y;
    t.z = v.y * p.z - u.y * q.z;

    t = mul(t, WorldToObject3x4()).xyz;

    float3 b;
    b.x = u.x * q.x - v.x * p.x;
    b.y = u.x * q.y - v.x * p.y;
    b.z = u.x * q.z - v.x * p.z;

    b = mul(b, WorldToObject3x4()).xyz;

    // orthogonalize T and B to N
    t = t - hit.nrm * dot(t, hit.nrm);
    b = b - hit.nrm * dot(b, hit.nrm);

    hit.tangent   = normalize(t);
    hit.bitangent = normalize(b);

    hit.bitangentSign = dot(cross(hit.nrm, hit.tangent), hit.bitangent) > 0 ? -1.0 : 1.0;
  }
}

HitState GetHitState(GltfRenderPrimitive renderPrim, float bitangentFlip, float2 attribs)
{
  HitState hit;

  // Barycentric coordinate on the triangle
  float3 barycentrics = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uint3 triangleIndex = getTriangleIndices(renderPrim, PrimitiveIndex());

  // Position
  const float3 pos0     = getVertexPosition(renderPrim, triangleIndex.x);
  const float3 pos1     = getVertexPosition(renderPrim, triangleIndex.y);
  const float3 pos2     = getVertexPosition(renderPrim, triangleIndex.z);
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos               = mul(float4(position, 1.0), ObjectToWorld4x3());

  // Normal
  const float3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3       worldGeoNormal = normalize(mul(geoNormal, WorldToObject3x4()).xyz);
  hit.geonrm                  = worldGeoNormal;

  hit.nrm = worldGeoNormal;
  if(hasVertexNormal(renderPrim))
  {
    const float3 normal      = getInterpolatedVertexNormal(renderPrim, triangleIndex, barycentrics);
    float3       worldNormal = normalize(mul(normal, WorldToObject3x4()).xyz);
    adjustShadingNormalToRayDir(worldNormal, worldGeoNormal);
    hit.nrm = worldNormal;
  }

  // TexCoord
  hit.uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

  // Tangent - Bitangent
  if(hasVertexTangent(renderPrim))
  {
    float4 tng[3];
    tng[0] = getVertexTangent(renderPrim, triangleIndex.x);
    tng[1] = getVertexTangent(renderPrim, triangleIndex.y);
    tng[2] = getVertexTangent(renderPrim, triangleIndex.z);

    hit.tangent   = normalize(mixBary(tng[0].xyz, tng[1].xyz, tng[2].xyz, barycentrics));  // interpolate tangent
    hit.tangent   = mul(hit.tangent, WorldToObject3x4()).xyz;                              // transform to worldspace
    hit.tangent   = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));  // orthogonalize to N and normalize
    hit.bitangent = cross(hit.nrm, hit.tangent) * tng[0].w;
    hit.bitangentSign = tng[0].w;
  }
  else
  {
    computeTangentSpace(renderPrim, triangleIndex, hit);
  }

  hit.bitangentSign *= bitangentFlip;
  hit.bitangent *= bitangentFlip;

  return hit;
}


#endif
