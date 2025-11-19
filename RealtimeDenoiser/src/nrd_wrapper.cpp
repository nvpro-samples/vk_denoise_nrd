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

#include "nrd_wrapper.hpp"

#include <nvvk/command_pools.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/shaders.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/default_structs.hpp>

#include <NRDDescs.h>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cassert>
#include <vector>
#include <sstream>
#include <stddef.h>
#include <stdint.h>

#define CALL_NRD(x)                                                                                                    \
  {                                                                                                                    \
    nrd::Result res = x;                                                                                               \
    assert(res == nrd::Result::SUCCESS && #x);                                                                         \
  }

#define NRD_ARRAYSIZE(x) (sizeof(x) / sizeof(*x))

// Translate NRD format enum values to Vulkan formats
static const VkFormat g_NRDFormatToVkFormat[] = {
    VK_FORMAT_R8_UNORM,
    VK_FORMAT_R8_SNORM,
    VK_FORMAT_R8_UINT,
    VK_FORMAT_R8_SINT,

    VK_FORMAT_R8G8_UNORM,
    VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT,
    VK_FORMAT_R8G8_SINT,

    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R8G8B8A8_SNORM,
    VK_FORMAT_A8B8G8R8_UINT_PACK32,
    VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R8G8B8A8_SRGB,

    VK_FORMAT_R16_UNORM,
    VK_FORMAT_R16_SNORM,
    VK_FORMAT_R16_UINT,
    VK_FORMAT_R16_SINT,
    VK_FORMAT_R16_SFLOAT,

    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT,
    VK_FORMAT_R16G16_SINT,
    VK_FORMAT_R16G16_SFLOAT,

    VK_FORMAT_R16G16B16A16_UNORM,
    VK_FORMAT_R16G16B16A16_SNORM,
    VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT,

    VK_FORMAT_R32_UINT,
    VK_FORMAT_R32_SINT,
    VK_FORMAT_R32_SFLOAT,

    VK_FORMAT_R32G32_UINT,
    VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R32G32_SFLOAT,

    VK_FORMAT_R32G32B32_UINT,
    VK_FORMAT_R32G32B32_SINT,
    VK_FORMAT_R32G32B32_SFLOAT,

    VK_FORMAT_R32G32B32A32_UINT,
    VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_R32G32B32A32_SFLOAT,

    VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    VK_FORMAT_A2R10G10B10_UINT_PACK32,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
};

static_assert(NRD_ARRAYSIZE(g_NRDFormatToVkFormat) == size_t(nrd::Format::MAX_NUM));

static inline VkFormat NRDtoVKFormat(nrd::Format nrdFormat)
{
  assert(size_t(nrdFormat) < NRD_ARRAYSIZE(g_NRDFormatToVkFormat));
  return g_NRDFormatToVkFormat[size_t(nrdFormat)];
}

// Translate NRD descriptor types to Vulkan descriptor types
static const VkDescriptorType g_NRDDescriptorTypeToVulkan[] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
static_assert(NRD_ARRAYSIZE(g_NRDDescriptorTypeToVulkan) == size_t(nrd::DescriptorType::MAX_NUM));

static inline VkDescriptorType NRDDescriptorTypeToVulkan(nrd::DescriptorType type)
{
  assert(size_t(type) < size_t(nrd::DescriptorType::MAX_NUM));
  return g_NRDDescriptorTypeToVulkan[size_t(type)];
}

// Translate NRD filters enum to Vulkan texture filters
static const VkFilter g_NRDtoVkFilter[] = {VK_FILTER_NEAREST, VK_FILTER_LINEAR};
static_assert(NRD_ARRAYSIZE(g_NRDtoVkFilter) == size_t(nrd::Sampler::MAX_NUM));

static inline VkFilter NRDtoVkFilter(nrd::Sampler sampler)
{
  assert(size_t(sampler) < NRD_ARRAYSIZE(g_NRDtoVkFilter));
  return g_NRDtoVkFilter[size_t(sampler)];
}

static inline uint16_t DivideRoundUp(uint32_t dividend, uint16_t divisor)
{
  return uint16_t((dividend + divisor - 1) / divisor);
}


NRDWrapper::NRDWrapper(nvvk::ResourceAllocator& alloc,
                       const nvvk::QueueInfo&   queue,
                       nvvk::SamplerPool&       samplerPool,
                       uint16_t                 width,
                       uint16_t                 height,
                       const nvvk::Image        userTexturePool[size_t(nrd::ResourceType::MAX_NUM)])
    : m_device(alloc.getDevice())
    , m_queue(queue)
    , m_samplerPool(samplerPool)
    , m_resAlloc(alloc)
{
  // NRDWrapper currently only exposes REBLUR_DIFFUSE_SPECULAR and RELAX_DIFFUSE_SPECULAR denoisers.
  // We directly use the nrd::Denoiser enum as 'identifier'.
  std::vector<nrd::DenoiserDesc> denoisers{
      {nrd::Identifier(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR},
      {nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR), nrd::Denoiser::RELAX_DIFFUSE_SPECULAR},
      // Have two separate reference denoisers, so we can denoise diffuse and specular separately
      {nrd::Identifier(nrd::Denoiser::REFERENCE), nrd::Denoiser::REFERENCE},
      {nrd::Identifier(nrd::Denoiser::REFERENCE) + 1, nrd::Denoiser::REFERENCE},
  };

  nrd::InstanceCreationDesc instanceDesc{{}, denoisers.data(), uint32_t(denoisers.size())};

  CALL_NRD(CreateInstance(instanceDesc, m_instance));

  // Query the Denoiser instance for its required resources and create them
  const nrd::InstanceDesc* iDesc = GetInstanceDesc(*m_instance);

  // Create the pool of permanent textures
  for(uint32_t t = 0; t < iDesc->permanentPoolSize; ++t)
  {
    nvvk::Image nrdTexture = createTexture(iDesc->permanentPool[t], width, height);
    m_permanentTextures.push_back(nrdTexture);

    std::stringstream name;
    name << "NRD_PermanentPool " << t;
    nvvk::DebugUtil::getInstance().setObjectName(nrdTexture.image, name.str());
    nvvk::DebugUtil::getInstance().setObjectName(nrdTexture.descriptor.imageView, name.str());
  }

  /* Create the pool of transient textures. It would be possible to
   * the application to reuse or alias these textures and their memory outside of the denoiser
   * but we don't make use of that here.
   */
  for(uint32_t t = 0; t < iDesc->transientPoolSize; ++t)
  {
    nvvk::Image nrdTexture = createTexture(iDesc->transientPool[t], width, height);
    m_transientTextures.push_back(nrdTexture);

    std::stringstream name;
    name << "NRD_TransientPool " << t;
    nvvk::DebugUtil::getInstance().setObjectName(nrdTexture.image, name.str());
    nvvk::DebugUtil::getInstance().setObjectName(nrdTexture.descriptor.imageView, name.str());
  }

  // Make a copy of the user texture pool
  for(uint32_t t = 0; t < uint32_t(nrd::ResourceType::MAX_NUM); ++t)
  {
    m_userTexturePool[t] = userTexturePool[t];
  }

  // Change color image layout and clear the images
  {
    const VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

    VkCommandPool   cmdPool = nvvk::createTransientCommandPool(m_device, m_queue.familyIndex);
    VkCommandBuffer cmd     = nvvk::createSingleTimeCommands(m_device, cmdPool);

    auto transitionTexture = [cmd, layout](VkImage image) {
      nvvk::cmdImageMemoryBarrier(cmd, {.image = image, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = layout});

      // Clear to avoid garbage data
      VkClearColorValue       clear_value = {{0.F, 0.F, 0.F, 0.F}};
      VkImageSubresourceRange range       = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cmd, image, layout, &clear_value, 1, &range);
    };

    for(auto& t : m_transientTextures)
    {
      transitionTexture(t.image);
      t.descriptor.imageLayout = layout;
    }
    for(auto& t : m_permanentTextures)
    {
      transitionTexture(t.image);
      t.descriptor.imageLayout = layout;
    }

    NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, cmdPool, m_queue.queue));
    vkDestroyCommandPool(m_device, cmdPool, nullptr);
  }

  // Create the samplers
  for(uint32_t s = 0; s < iDesc->samplersNum; ++s)
  {
    auto                filter = NRDtoVkFilter(iDesc->samplers[s]);
    VkSamplerCreateInfo sInfo  = DEFAULT_VkSamplerCreateInfo;
    sInfo.minFilter            = filter;
    sInfo.magFilter            = filter;
    sInfo.addressModeU = sInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sInfo.mipmapMode                        = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    VkSampler sampler;

    NVVK_CHECK(m_samplerPool.acquireSampler(sampler, sInfo));
    m_samplers.push_back(sampler);
  }

  // Create the constant buffer
  NVVK_CHECK(m_resAlloc.createBuffer(m_constantBuffer, VkDeviceSize(iDesc->constantBufferMaxDataSize),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));

  createPipelines();
}

NRDWrapper::~NRDWrapper()
{
  vkDeviceWaitIdle(m_device);

  m_resAlloc.destroyBuffer(m_constantBuffer);
  for(auto s : m_samplers)
  {
    m_samplerPool.releaseSampler(s);
  }
  for(auto& t : m_transientTextures)
  {
    m_resAlloc.destroyImage(t);
  }
  for(auto& t : m_permanentTextures)
  {
    m_resAlloc.destroyImage(t);
  }
  if(m_samplerConstBufferDescriptorLayout)  // If samplers were in a different set
  {
    vkDestroyDescriptorSetLayout(m_device, m_samplerConstBufferDescriptorLayout, nullptr);
    vkDestroyDescriptorPool(m_device, m_samplerConstBufferDescriptorPool, nullptr);
  }
  for(auto& p : m_pipelines)
  {
    vkDestroyPipeline(m_device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, p.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, p.resourceDescriptorLayout, nullptr);
  }

  nrd::DestroyInstance(*m_instance);
}

void NRDWrapper::setUserPoolTexture(nrd::ResourceType resource, nvvk::Image texture)
{
  m_userTexturePool[size_t(resource)] = texture;
}

VkFormat NRDWrapper::getNormalRoughnessFormat()
{
  // The NRD library can be compiled with different kinds of normal encodings
  // in mind. We have to chose accordingly.
  switch(nrd::GetLibraryDesc()->normalEncoding)
  {
    case nrd::NormalEncoding::RGBA8_UNORM:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case nrd::NormalEncoding::RGBA8_SNORM:
      return VK_FORMAT_R8G8B8A8_SNORM;
    case nrd::NormalEncoding::R10_G10_B10_A2_UNORM:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case nrd::NormalEncoding::RGBA16_UNORM:
      return VK_FORMAT_R16G16B16A16_UNORM;
    case nrd::NormalEncoding::RGBA16_SNORM:
      return VK_FORMAT_R16G16B16A16_SNORM;
      // NRD documentation says RGBA16_SNORM may also translate to a floating point format
      //return VK_FORMAT_R16G16B16A16_SFLOAT;
    default:
      assert(0 && "Unknown normal encoding");
  }
  return VK_FORMAT_UNDEFINED;
}

void NRDWrapper::setCommonSettings(nrd::CommonSettings& settings)
{
  CALL_NRD(nrd::SetCommonSettings(*m_instance, settings));
}


void NRDWrapper::setDenoiserSettings(nrd::Identifier identifier, const void* settings)
{
  CALL_NRD(nrd::SetDenoiserSettings(*m_instance, identifier, settings));
}

void NRDWrapper::setREBLURSettings(nrd::ReblurSettings const& settings)
{
  setDenoiserSettings(nrd::Identifier(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR), &settings);
}
void NRDWrapper::setRELAXSettings(const nrd::RelaxSettings& settings)
{
  setDenoiserSettings(nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR), &settings);
}

nvvk::Image NRDWrapper::createTexture(const nrd::TextureDesc& tDesc, uint16_t width, uint16_t height)
{

  uint16_t texWidth  = DivideRoundUp(width, tDesc.downsampleFactor);
  uint16_t texHeight = DivideRoundUp(height, tDesc.downsampleFactor);

  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.extent            = {texWidth, texHeight, 1};
  imageInfo.format            = NRDtoVKFormat(tDesc.format);
  imageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

  nvvk::Image image;

  NVVK_CHECK(m_resAlloc.createImage(image, imageInfo, DEFAULT_VkImageViewCreateInfo));

  assert(image.image && image.descriptor.imageView);
  return image;
}

// NRD provides us with a description of all involved pipelines.
// The pipeline shader code has been precompiled to SPIR_V as part of the NRD library build.
// The NRD shaders were written with specific expectations about which descriptor sets
// contain which texture/sampler/buffer binding indices.
// Here we need to build pipeline layouts that exactly reconstruct these bindings.
// Then we build compute pipelines using the provided binary shader code and the layouts
// created from NRD's descriptions.
void NRDWrapper::createPipelines()
{
  const nrd::InstanceDesc* iDesc = nrd::GetInstanceDesc(*m_instance);
  const nrd::LibraryDesc*  lDesc = nrd::GetLibraryDesc();

  // These are the base binding indices for each type of binding
  const uint32_t constantBufferBindingOffset   = lDesc->spirvBindingOffsets.constantBufferOffset;
  const uint32_t samplersBindingOffset         = lDesc->spirvBindingOffsets.samplerOffset;
  const uint32_t texturesBindingOffset         = lDesc->spirvBindingOffsets.textureOffset;
  const uint32_t storageTextureAndBufferOffset = lDesc->spirvBindingOffsets.storageTextureAndBufferOffset;

  // If samplers are in a separate descriptor set, create a descriptor set for
  // them now.
  // If NRD placed samplers in a separate descriptor set, create it now.
  bool samplersInSeparateSet = iDesc->constantBufferAndSamplersSpaceIndex != iDesc->resourcesSpaceIndex;
  assert(samplersInSeparateSet);

  {
    // Prepare immutable sampler descriptors
    std::vector<VkDescriptorSetLayoutBinding> setBindings(iDesc->samplersNum + 1);
    for(uint32_t s = 0; s < iDesc->samplersNum; ++s)
    {
      setBindings[s] = VkDescriptorSetLayoutBinding{.binding         = samplersBindingOffset + s,
                                                    .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER,
                                                    .descriptorCount = 1,
                                                    .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
                                                    // We make use of immutable samplers as they will be shared among all the pipelines and
                                                    // don't change over the lifetime of the pipelines
                                                    .pImmutableSamplers = &m_samplers[s]};
    }
    // Prepare constant buffer descriptor
    setBindings[iDesc->samplersNum] = VkDescriptorSetLayoutBinding{
        .binding         = constantBufferBindingOffset,
        .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo samplerLayoutInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                            .flags        = 0,
                                                            .bindingCount = (uint32_t)setBindings.size(),
                                                            .pBindings    = setBindings.data()};
    NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &samplerLayoutInfo, nullptr, &m_samplerConstBufferDescriptorLayout));

    const VkDescriptorPoolSize poolSizes[] = {{.type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = iDesc->samplersNum},
                                              {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1}};

    const VkDescriptorPoolCreateInfo poolInfo{.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                              .maxSets       = 1,
                                              .poolSizeCount = NRD_ARRAYSIZE(poolSizes),
                                              .pPoolSizes    = poolSizes};
    NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_samplerConstBufferDescriptorPool));

    const VkDescriptorSetAllocateInfo descriptorSetInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                        .descriptorPool     = m_samplerConstBufferDescriptorPool,
                                                        .descriptorSetCount = 1,
                                                        .pSetLayouts        = &m_samplerConstBufferDescriptorLayout};

    NVVK_CHECK(vkAllocateDescriptorSets(m_device, &descriptorSetInfo, &m_samplerConstBufferDescriptorSet));
  }

  // Bind the constant buffer once and leave it there
  {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_constantBuffer.buffer;
    bufferInfo.offset = 0;
    bufferInfo.range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet constantBufferUpdate = {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                 .pNext           = nullptr,
                                                 .dstSet          = m_samplerConstBufferDescriptorSet,
                                                 .dstBinding      = constantBufferBindingOffset,
                                                 .descriptorCount = 1,
                                                 .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                 .pBufferInfo     = &bufferInfo};
    vkUpdateDescriptorSets(m_device, 1, &constantBufferUpdate, 0, nullptr);
  }

  // Determine the maximum number of bindings a pipeline can have
  uint32_t maxNumTextureBindings = 0;
  for(uint32_t p = 0; p < iDesc->pipelinesNum; ++p)
  {
    const nrd::PipelineDesc& nrdPipelineDesc = iDesc->pipelines[p];

    uint32_t numResources = 0;
    for(uint32_t r = 0; r < nrdPipelineDesc.resourceRangesNum; ++r)
    {
      numResources += nrdPipelineDesc.resourceRanges[r].descriptorsNum;
    }
    maxNumTextureBindings = std::max(maxNumTextureBindings, numResources);
  }

  m_pipelines.resize(iDesc->pipelinesNum);

#pragma omp parallel for
  for(int p = 0; p < static_cast<int>(iDesc->pipelinesNum); ++p)
  {
    LOGI("Compiling NRD pipeline %d\n", p);

    const nrd::PipelineDesc& pDesc = iDesc->pipelines[p];

    // Just reserve maximum number of bindings - we may not make use of them all for each pipeline
    // m textures (either for sampling or as storage)
    std::vector<VkDescriptorSetLayoutBinding> setBindings(maxNumTextureBindings);
    // On the main descriptor set, we make use of push descriptors which makes it so much easier to use and update.
    // But we can't use push descriptors for both; see VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293
    VkDescriptorSetLayoutCreateInfo setLayoutInfo{.sType     = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                  .flags     = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                  .pBindings = setBindings.data()};

    // Prepare image/texture descriptors
    for(uint32_t r = 0; r < pDesc.resourceRangesNum; ++r)
    {
      const nrd::ResourceRangeDesc& range = pDesc.resourceRanges[r];

      for(uint32_t b = 0; b < range.descriptorsNum; ++b)
      {
        VkDescriptorSetLayoutBinding binding{.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};

        switch(range.descriptorType)
        {
          case nrd::DescriptorType::TEXTURE:
            binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            binding.binding        = texturesBindingOffset + iDesc->resourcesBaseRegisterIndex + b;
            break;
          case nrd::DescriptorType::STORAGE_TEXTURE:
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            binding.binding        = storageTextureAndBufferOffset + iDesc->resourcesBaseRegisterIndex + b;
            break;
          default:
            assert(0);
        }
        setBindings[setLayoutInfo.bindingCount++] = binding;
      }
    }

    // Now let's build the layouts, descriptor sets and pipelines
    NRDPipeline& nrdPipeline = m_pipelines[p];

    NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &setLayoutInfo, nullptr, &nrdPipeline.resourceDescriptorLayout));

    nrdPipeline.numBindings = setLayoutInfo.bindingCount;
    LOGI("Pipeline uses %d bindings\n", nrdPipeline.numBindings);

    // NRD using these two set indexes is a hardcoded assumption that NRD promised not to break
    assert(iDesc->constantBufferAndSamplersSpaceIndex == 1 && iDesc->resourcesSpaceIndex == 0);

    // Each pipeline is accessing the global sampler+constant buffer set as well as a pipeline-specific set of textures
    const std::array<VkDescriptorSetLayout, 2> pipelineSetLayouts{nrdPipeline.resourceDescriptorLayout,
                                                                  m_samplerConstBufferDescriptorLayout};
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                        .setLayoutCount = 2U,
                                                        .pSetLayouts    = pipelineSetLayouts.data()};

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &nrdPipeline.pipelineLayout));

    VkShaderModule computeShaderModule;
    NVVK_CHECK(nvvk::createShaderModule(computeShaderModule, m_device,
                                        {(const uint32_t*)pDesc.computeShaderSPIRV.bytecode, pDesc.computeShaderSPIRV.size / 4}));
    assert(computeShaderModule);

    const VkPipelineShaderStageCreateInfo stageCreateInfo{.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                          .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                                                          .module = computeShaderModule,
                                                          .pName  = "main"};

    const VkComputePipelineCreateInfo pipelineCreateInfo{.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                         .stage  = stageCreateInfo,
                                                         .layout = nrdPipeline.pipelineLayout};

    NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &nrdPipeline.pipeline));

    vkDestroyShaderModule(m_device, computeShaderModule, nullptr);
  }
}

void NRDWrapper::denoise(const nrd::Identifier* denoisers, uint32_t denoisersNum, VkCommandBuffer& commandBuffer)
{
  const nrd::DispatchDesc* dispatchDescs    = nullptr;
  uint32_t                 dispatchDescsNum = 0;
  nrd::GetComputeDispatches(*m_instance, reinterpret_cast<const nrd::Identifier*>(denoisers), denoisersNum,
                            dispatchDescs, dispatchDescsNum);

  for(uint32_t d = 0; d < dispatchDescsNum; ++d)
  {
    const nrd::DispatchDesc& dDesc = dispatchDescs[d];

    nvvk::DebugUtil::ScopedCmdLabel cmdBufLabel(commandBuffer, dDesc.name);

    dispatch(commandBuffer, dDesc);
  }
}

// NRD provides us with a description of which image it wants to bind to which
// descriptor binding index.
void NRDWrapper::dispatch(VkCommandBuffer commandBuffer, const nrd::DispatchDesc& dispatchDesc)
{
  const nrd::LibraryDesc*  lDesc = nrd::GetLibraryDesc();
  const nrd::InstanceDesc* iDesc = nrd::GetInstanceDesc(*m_instance);
  const nrd::PipelineDesc& pDesc = iDesc->pipelines[dispatchDesc.pipelineIndex];

  // These are the base binding indices for each type of textures
  const uint32_t texturesBindingOffset         = lDesc->spirvBindingOffsets.textureOffset;
  const uint32_t storageTextureAndBufferOffset = lDesc->spirvBindingOffsets.storageTextureAndBufferOffset;

  NRDPipeline& pipeline = m_pipelines[dispatchDesc.pipelineIndex];

  std::vector<VkWriteDescriptorSet>  descriptorUpdates(pipeline.numBindings + iDesc->samplersNum,
                                                       VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr});
  std::vector<VkDescriptorImageInfo> descriptorImageInfos(pipeline.numBindings + iDesc->samplersNum);

  std::vector<VkImageMemoryBarrier> imageBarriers;

  auto transitionToShaderRead = [](VkImage image) {
    return nvvk::makeImageMemoryBarrier({
        .image         = image,
        .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .srcStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    });
  };
  auto transitionToShaderWrite = [](VkImage image) {
    return nvvk::makeImageMemoryBarrier({.image         = image,
                                         .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
                                         .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
                                         .srcStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         .dstStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                         .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT});
  };

  // Determine texture descriptor set updates and corresponding image layout transitions
  uint32_t numResourceUpdates = 0;  // Count and index the updates
  for(uint32_t r = 0; r < pDesc.resourceRangesNum; ++r)
  {
    const nrd::ResourceRangeDesc& resourceRange = pDesc.resourceRanges[r];
    const bool                    isStorage     = resourceRange.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;

    uint32_t rangeBaseBindingIndex = isStorage ? storageTextureAndBufferOffset : texturesBindingOffset;

    for(uint32_t d = 0; d < resourceRange.descriptorsNum; ++d)
    {
      const nrd::ResourceDesc& nrdResource = dispatchDesc.resources[numResourceUpdates];

      VkWriteDescriptorSet& update = descriptorUpdates[numResourceUpdates];
      update.dstBinding            = rangeBaseBindingIndex + d;
      update.descriptorCount       = 1;
      update.descriptorType        = NRDDescriptorTypeToVulkan(nrdResource.descriptorType);
      update.pImageInfo            = &descriptorImageInfos[numResourceUpdates];

      assert(nrdResource.descriptorType == resourceRange.descriptorType);

      nvvk::Image* texture = nullptr;
      if(nrdResource.type == nrd::ResourceType::TRANSIENT_POOL)
      {
        texture = &m_transientTextures[nrdResource.indexInPool];
      }
      else if(nrdResource.type == nrd::ResourceType::PERMANENT_POOL)
      {
        texture = &m_permanentTextures[nrdResource.indexInPool];
      }
      else
      {
        texture = &m_userTexturePool[(uint32_t)nrdResource.type];
      }

      assert(texture->image && texture->descriptor.imageView);

      // We assume, images bound to storage bindings will be written to, while images bound to
      // texture bindings will be read from.
      // This is a rather simple scheme. If it turns out, these barriers cost too much performance,
      // we might want to be more clever about it by caching transitions between pipelines.
      isStorage ? transitionToShaderWrite(texture->image) : transitionToShaderRead(texture->image);

      descriptorImageInfos[numResourceUpdates] = texture->descriptor;
      ++numResourceUpdates;
    }
  }
  // Transition all resources into their appropriate state
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, (uint32_t)imageBarriers.size(), imageBarriers.data());

  const bool samplersInSeparateSet = iDesc->constantBufferAndSamplersSpaceIndex != iDesc->resourcesSpaceIndex;
  assert(samplersInSeparateSet);

  if(pDesc.hasConstantData)
  {
    if(!dispatchDesc.constantBufferDataMatchesPreviousDispatch)
    {
      {
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                         nullptr,
                                         VK_ACCESS_SHADER_READ_BIT,
                                         VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         m_constantBuffer.buffer,
                                         0,
                                         VK_WHOLE_SIZE};

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 1, &barrier, 0, nullptr);
      }

      vkCmdUpdateBuffer(commandBuffer, m_constantBuffer.buffer, 0, dispatchDesc.constantBufferDataSize, dispatchDesc.constantBufferData);

      {
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                         nullptr,
                                         VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_ACCESS_SHADER_READ_BIT,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         m_constantBuffer.buffer,
                                         0,
                                         VK_WHOLE_SIZE};

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 1, &barrier, 0, nullptr);
      }
    }
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

  // Bind the global set with the immutable samplers and constant buffer
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout,
                          iDesc->constantBufferAndSamplersSpaceIndex, 1, &m_samplerConstBufferDescriptorSet, 0, nullptr);

  // Update the texture descriptors. Notice how push descriptors don't require us to make sure the
  // descriptors are not in use anymore.
  vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout,
                            iDesc->resourcesSpaceIndex, numResourceUpdates, descriptorUpdates.data());

  // Go!
  vkCmdDispatch(commandBuffer, dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
}
