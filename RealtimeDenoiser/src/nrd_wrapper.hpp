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

#pragma once

#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/pipeline.hpp>
#include <nvvk/compute_pipeline.hpp>

#include <NRD.h>

#include <cstddef>
#include <cstdint>

class NRDWrapper
{
public:
  /* Create the Vulkan NRD Wrapper
   * NRDWrapper does not automatically resize its resources and will have to get recreated if the size changes.
   * However, it does support rendering to only a part of the images, as described in the NRD documentation
   * regarding nrd::CommonSettings::resourceSize and nrd::CommonSettings::rectSize.
   *
   * The userTexturePool is a pool of textures that NRD uses as input and output data.
   * Which textures are needed depends on the actual denoiser in use. Refer to NRDDescs.h
   * to find out which textures are needed for which denoiser.
   * To make it an easier interface, we use an array where each slot corresponds to one 'nrd::ResourceType'
   * texture resource. Depending on the denoiser in use, this array will be sparsely populated.
   * 'userTexturePool' (but NOT the textures it contains) will be copied into an internal copy
   * and thus can be discarded after the call.
   *
   * NRD uses two internal pools of textures ("resources"): permanent and transient ones.
   * Permanent textures must not be altered outside of NRD, while transient textures could be
   * reused as (or aliased with) other application specific textures. Albeit, this wrapper
   * does not expose the transient pool to the application and thus makes no use of reusing
   * transient textures for other purposes.
   */
  NRDWrapper(nvvk::ResourceAllocator& alloc,
             const nvvk::QueueInfo&   queue,
             nvvk::SamplerPool&       samplerPool,
             uint16_t                 width,
             uint16_t                 height,
             const nvvk::Image        userTexturePool[size_t(nrd::ResourceType::MAX_NUM)]);
  ~NRDWrapper();

  void setUserPoolTexture(nrd::ResourceType resource, nvvk::Image texture);

  /* Set common NRD settings, typically called once per frame */
  void setCommonSettings(nrd::CommonSettings& settings);

  /* Denoiser specifc settings */
  void setREBLURSettings(const nrd::ReblurSettings& ssettings);
  void setRELAXSettings(const nrd::RelaxSettings& settings);

  /* Perform the actual denoising. NRD will read from a number of 'IN_*' images in the user texture pool
   * and write to the 'OUT_' images specified by the denoiser.
   * Refer to NRDDescs.h for the per-denoiser input and output textures.
 */
  void denoise(const nrd::Identifier* denoisers, uint32_t denoisersNum, VkCommandBuffer& commandBuffer);

  /* When the NRD library is compiled, it is hardcoded to a specific Normal/Roughness encoding.
   * It requires to use a specific image format to store the encoded values.
   */
  static VkFormat getNormalRoughnessFormat();

private:
  struct NRDPipeline
  {
    VkPipeline            pipeline                 = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout           = VK_NULL_HANDLE;
    VkDescriptorSetLayout resourceDescriptorLayout = VK_NULL_HANDLE;
    uint32_t              numBindings              = 0;
  };

  NRDWrapper(const NRDWrapper&)           = delete;
  NRDWrapper& operator=(const NRDWrapper) = delete;

  nvvk::Image createTexture(const nrd::TextureDesc& tDesc, uint16_t width, uint16_t height);
  void        createPipelines();
  void        setDenoiserSettings(nrd::Identifier identifier, const void* settings);
  void        dispatch(VkCommandBuffer commandBuffer, const nrd::DispatchDesc& dispatchDesc);

  nrd::Instance*           m_instance = nullptr;
  VkDevice                 m_device   = VK_NULL_HANDLE;
  nvvk::QueueInfo          m_queue;
  nvvk::SamplerPool&       m_samplerPool;
  nvvk::ResourceAllocator& m_resAlloc;

  std::vector<nvvk::Image>                                    m_permanentTextures;
  std::vector<nvvk::Image>                                    m_transientTextures;
  std::array<nvvk::Image, size_t(nrd::ResourceType::MAX_NUM)> m_userTexturePool;
  std::vector<VkSampler>                                      m_samplers;
  nvvk::Buffer                                                m_constantBuffer;

  std::vector<NRDPipeline> m_pipelines;
  // Vulkan doesn't let us create a pipeline layout with two sets that use
  // push descriptors. So if NRD places immutable samplers in a different
  // set, then we store its set (which does not require updates) here.
  // We use a single one that will be shared among all pipelines.
  VkDescriptorSetLayout m_samplerConstBufferDescriptorLayout;
  VkDescriptorPool      m_samplerConstBufferDescriptorPool;
  VkDescriptorSet       m_samplerConstBufferDescriptorSet;
};
