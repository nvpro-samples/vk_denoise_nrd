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
mat3x2 getTexCoords0(in RenderPrimitive renderPrim, in uvec3 idx)
{
  if(!hasVertexTexCoord0(renderPrim))
    return mat3x2(0);

  VertexTexCoord0 texcoords = VertexTexCoord0(renderPrim.vertexBuffer.texCoord0Address);
  mat3x2            uv;
  uv[0] = texcoords._[idx.x];
  uv[1] = texcoords._[idx.y];
  uv[2] = texcoords._[idx.z];
  return uv;
}

vec2 getInterpolatedVertexTexCoords(in RenderPrimitive renderPrim, in uvec3 idx, in vec3 barycentrics)
{
  if(!hasVertexTexCoord0(renderPrim))
    return vec2(0, 0);

  mat3x2 uv = getTexCoords0(renderPrim, idx);

  return uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;
}


void computeTangentSpace(in RenderPrimitive renderPrim, in uvec3 idx, inout HitState hit)
{
  mat3x2 uv = getTexCoords0(renderPrim, idx);

  vec2 u = uv[1] - uv[0];
  vec2 v = uv[2] - uv[0];

  float d = u.x * v.y - u.y * v.x;
  if (d == 0.0f)
  {
    vec4 t = makeFastTangent(hit.nrm);
    hit.tangent = t.xyz;
    hit.bitangent = cross(hit.nrm, hit.tangent) * t.w;
    hit.bitangentSign = t.w;
  }
  else
  {
    u /= d;
    v /= d;

    vec3 v0 = getVertexPosition(renderPrim, idx.x);
    vec3 v1 = getVertexPosition(renderPrim, idx.y);
    vec3 v2 = getVertexPosition(renderPrim, idx.z);

    vec3 p = v1 - v0;
    vec3 q = v2 - v0;

    vec3 t;
    t.x = v.y * p.x - u.y * q.x;
    t.y = v.y * p.y - u.y * q.y;
    t.z = v.y * p.z - u.y * q.z;

    t = vec3(t * gl_WorldToObjectEXT);

    vec3 b;
    b.x = u.x * q.x - v.x * p.x;
    b.y = u.x * q.y - v.x * p.y;
    b.z = u.x * q.z - v.x * p.z;

    b = vec3(b * gl_WorldToObjectEXT);

    // orthogonalize T and B to N
    t = t - hit.nrm * dot(t, hit.nrm);
    b = b - hit.nrm * dot(b, hit.nrm);

    hit.tangent = normalize(t);
    hit.bitangent = normalize(b);

    hit.bitangentSign = dot(cross(hit.nrm, hit.tangent), hit.bitangent) > 0 ? -1.0 : 1.0;
  }
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
HitState GetHitState(RenderPrimitive renderPrim, in float bitangentFlip)
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
  if(hasVertexTangent(renderPrim))
  {
    vec4 tng[3];
    tng[0] = getVertexTangent(renderPrim, triangleIndex.x);
    tng[1] = getVertexTangent(renderPrim, triangleIndex.y);
    tng[2] = getVertexTangent(renderPrim, triangleIndex.z);

    hit.tangent   = normalize(mixBary(tng[0].xyz, tng[1].xyz, tng[2].xyz, barycentrics)); // interpolate tangent
    hit.tangent   = vec3(hit.tangent * gl_WorldToObjectEXT); // transform to worldspace
    hit.tangent   = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent)); // orthogonalize to N and normalize
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
