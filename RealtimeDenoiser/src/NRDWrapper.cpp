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

#include "NRDWrapper.hpp"

#include <nvvk/images_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvh/nvprint.hpp>
#include <nvvk/commands_vk.hpp>

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

#define ARRAYSIZE(x) (sizeof(x) / sizeof(*x))

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

static_assert(ARRAYSIZE(g_NRDFormatToVkFormat) == size_t(nrd::Format::MAX_NUM));

static inline VkFormat NRDtoVKFormat(nrd::Format nrdFormat)
{
  assert(size_t(nrdFormat) < ARRAYSIZE(g_NRDFormatToVkFormat));
  return g_NRDFormatToVkFormat[size_t(nrdFormat)];
}

// Translate NRD descriptor types to Vulkan descriptor types
static const VkDescriptorType g_NRDDescriptorTypeToVulkan[] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
static_assert(ARRAYSIZE(g_NRDDescriptorTypeToVulkan) == size_t(nrd::DescriptorType::MAX_NUM));

static inline VkDescriptorType NRDDescriptorTypeToVulkan(nrd::DescriptorType type)
{
  assert(size_t(type) < size_t(nrd::DescriptorType::MAX_NUM));
  return g_NRDDescriptorTypeToVulkan[size_t(type)];
}

// Translate NRD filters enum to Vulkan texture filters
static const VkFilter g_NRDtoVkFilter[] = {VK_FILTER_NEAREST, VK_FILTER_LINEAR};
static_assert(ARRAYSIZE(g_NRDtoVkFilter) == size_t(nrd::Sampler::MAX_NUM));

static inline VkFilter NRDtoVkFilter(nrd::Sampler sampler)
{
  assert(size_t(sampler) < ARRAYSIZE(g_NRDtoVkFilter));
  return g_NRDtoVkFilter[size_t(sampler)];
}

static inline uint16_t DivideRoundUp(uint32_t dividend, uint16_t divisor)
{
  return uint16_t((dividend + divisor - 1) / divisor);
}


NRDWrapper::NRDWrapper(nvvk::ResourceAllocator& alloc,
                       uint16_t                 width,
                       uint16_t                 height,
                       const nvvk::Texture      userTexturePool[size_t(nrd::ResourceType::MAX_NUM)])
    : m_device(alloc.getDevice())
    , m_resAlloc(alloc)
    , m_dbgUtil(m_device)
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
  nrd::InstanceDesc iDesc = GetInstanceDesc(*m_instance);

  // Create the pool of permanent textures
  for(uint32_t t = 0; t < iDesc.permanentPoolSize; ++t)
  {
    nvvk::Texture nrdTexture = createTexture(iDesc.permanentPool[t], width, height);
    m_permanentTextures.push_back(nrdTexture);

    std::stringstream name;
    name << "NRD_PermanentPool " << t;
    m_dbgUtil.setObjectName(nrdTexture.image, name.str());
    m_dbgUtil.setObjectName(nrdTexture.descriptor.imageView, name.str());
  }

  /* Create the pool of transient textures. It would be possible to
   * the application to reuse or alias these textures and their memory outside of the denoiser
   * but we don't make use of that here.
   */
  for(uint32_t t = 0; t < iDesc.transientPoolSize; ++t)
  {
    nvvk::Texture nrdTexture = createTexture(iDesc.transientPool[t], width, height);
    m_transientTextures.push_back(nrdTexture);

    std::stringstream name;
    name << "NRD_TransientPool " << t;
    m_dbgUtil.setObjectName(nrdTexture.image, name.str());
    m_dbgUtil.setObjectName(nrdTexture.descriptor.imageView, name.str());
  }

  // Make a copy of the user texture pool
  for(uint32_t t = 0; t < uint32_t(nrd::ResourceType::MAX_NUM); ++t)
  {
    m_userTexturePool[t] = userTexturePool[t];
  }

  // Change color image layout and clear the images
  {
    const VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

    nvvk::CommandPool cpool(m_device, 0);
    VkCommandBuffer   cmd = cpool.createCommandBuffer();

    auto transitionTexture = [cmd, layout](VkImage image) {
      nvvk::cmdBarrierImageLayout(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, layout);

      // Clear to avoid garbage data
      VkClearColorValue       clear_value = {{0.F, 0.F, 0.F, 0.F}};
      VkImageSubresourceRange range       = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cmd, image, layout, &clear_value, 1, &range);
    };

    for(auto& t : m_transientTextures)
    {
      transitionTexture(t.image);
    }
    for(auto& t : m_permanentTextures)
    {
      transitionTexture(t.image);
    }

    cpool.submitAndWait(cmd);
  }

  // Create the samplers
  for(uint32_t s = 0; s < iDesc.samplersNum; ++s)
  {
    auto                filter = NRDtoVkFilter(iDesc.samplers[s]);
    VkSamplerCreateInfo sInfo =
        nvvk::makeSamplerCreateInfo(filter, filter, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_FALSE, 1.0f, VK_SAMPLER_MIPMAP_MODE_NEAREST);

    VkSampler sampler = alloc.acquireSampler(sInfo);
    m_samplers.push_back(sampler);
  }

  // Create the constant buffer
  m_constantBuffer = m_resAlloc.createBuffer(VkDeviceSize(iDesc.constantBufferMaxDataSize),
                                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  createPipelines();
}

NRDWrapper::~NRDWrapper()
{
  vkDeviceWaitIdle(m_device);

  m_resAlloc.destroy(m_constantBuffer);
  for(auto s : m_samplers)
  {
    m_resAlloc.releaseSampler(s);
  }
  for(auto& t : m_transientTextures)
  {
    m_resAlloc.destroy(t);
  }
  for(auto& t : m_permanentTextures)
  {
    m_resAlloc.destroy(t);
  }
  for(auto& p : m_pipelines)
  {
    vkDestroyPipeline(m_device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, p.pipelineLayout, nullptr);
    for(auto& l : p.descriptorLayouts)
    {
      vkDestroyDescriptorSetLayout(m_device, l, nullptr);
    }
  }

  nrd::DestroyInstance(*m_instance);
}

void NRDWrapper::setUserPoolTexture(nrd::ResourceType resource, nvvk::Texture texture)
{
  m_userTexturePool[size_t(resource)] = texture;
}

VkFormat NRDWrapper::getNormalRoughnessFormat()
{
  // The NRD library can be compiled with different kinds of normal encodings
  // in mind. We have to chose accordingly.
  switch(nrd::GetLibraryDesc().normalEncoding)
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

nvvk::Texture NRDWrapper::createTexture(const nrd::TextureDesc& tDesc, uint16_t width, uint16_t height)
{

  uint16_t texWidth  = DivideRoundUp(width, tDesc.downsampleFactor);
  uint16_t texHeight = DivideRoundUp(height, tDesc.downsampleFactor);

  VkImageCreateInfo     imgInfo  = nvvk::makeImage2DCreateInfo({texWidth, texHeight}, NRDtoVKFormat(tDesc.format),
                                                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, false);
  nvvk::Image           image    = m_resAlloc.createImage(imgInfo);
  VkImageViewCreateInfo viewInfo = nvvk::makeImageViewCreateInfo(image.image, imgInfo);
  nvvk::Texture         texture  = m_resAlloc.createTexture(image, viewInfo);

  assert(image.image && texture.descriptor.imageView);
  return texture;
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
  nrd::InstanceDesc iDesc = nrd::GetInstanceDesc(*m_instance);
  nrd::LibraryDesc  lDesc = nrd::GetLibraryDesc();

  // These are the base binding indices for each type of binding
  uint32_t constantBufferBindingOffset   = lDesc.spirvBindingOffsets.constantBufferOffset;
  uint32_t samplersBindingOffset         = lDesc.spirvBindingOffsets.samplerOffset;
  uint32_t resourcesBindingOffset        = lDesc.spirvBindingOffsets.textureOffset;
  uint32_t storageTextureAndBufferOffset = lDesc.spirvBindingOffsets.storageTextureAndBufferOffset;

  // Determine the number of unique sets ("register spaces")
  // The indices here store which type of resource goes into which set.
  // NRD can make it so that each type goes into its own set or sets are shared among resource types.
  m_constantBufferSetIndex = 0;
  m_samplersSetIndex       = (iDesc.constantBufferSpaceIndex == iDesc.samplersSpaceIndex) ? m_constantBufferSetIndex :
                                                                                            m_constantBufferSetIndex + 1;
  m_resourcesSetIndex =
      (iDesc.resourcesSpaceIndex == iDesc.constantBufferSpaceIndex) ?
          0 :
          ((iDesc.resourcesSpaceIndex == iDesc.samplersSpaceIndex) ? m_samplersSetIndex : m_samplersSetIndex + 1);
  uint32_t numPipelineSets = std::max(m_samplersSetIndex, m_resourcesSetIndex) + 1;

  // Determine the maximum number of bindings a pipeline can have
  uint32_t maxNumtextureBindings = 0;
  for(uint32_t p = 0; p < iDesc.pipelinesNum; ++p)
  {
    const nrd::PipelineDesc& nrdPipelineDesc = iDesc.pipelines[p];

    uint32_t numResources = 0;
    for(uint32_t r = 0; r < nrdPipelineDesc.resourceRangesNum; ++r)
    {
      numResources += nrdPipelineDesc.resourceRanges[r].descriptorsNum;
    }
    maxNumtextureBindings = std::max(maxNumtextureBindings, numResources);
  }

  m_pipelines.resize(iDesc.pipelinesNum);

  for(uint32_t p = 0; p < iDesc.pipelinesNum; ++p)
  {
    LOGI("Compiling NRD pipeline %d\n", p);

    const nrd::PipelineDesc& pDesc = iDesc.pipelines[p];

    std::vector<VkDescriptorSetLayoutCreateInfo> descriptorSetLayoutInfos(numPipelineSets,
                                                                          nvvk::make<VkDescriptorSetLayoutCreateInfo>());
    // We make use of push descriptors which makes it so much easier to use and update.
    for(auto& layout : descriptorSetLayoutInfos)
    {
      layout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }
    // Just reserve maximum number of bindings - we may not make use of them all for each pipeline
    // 1 constant buffer
    // n samplers
    // m textures
    std::vector<std::vector<VkDescriptorSetLayoutBinding>> setBindings(
        numPipelineSets, std::vector<VkDescriptorSetLayoutBinding>(1 + iDesc.samplersNum + maxNumtextureBindings));

    // Prepare sampler descriptors
    {
      VkDescriptorSetLayoutCreateInfo& samplerBindingSetInfo = descriptorSetLayoutInfos[m_constantBufferSetIndex];
      if(!samplerBindingSetInfo.pBindings)
      {
        // This path will only be hit when the samplers are in their own set and we just start putting sampler descriptors in there
        samplerBindingSetInfo.pBindings = setBindings[m_constantBufferSetIndex].data();
      }

      for(uint32_t s = 0; s < iDesc.samplersNum; ++s)
      {
        VkDescriptorSetLayoutBinding& samplerBindings =
            setBindings[m_constantBufferSetIndex][samplerBindingSetInfo.bindingCount++];

        samplerBindings.binding         = samplersBindingOffset + s;
        samplerBindings.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
        samplerBindings.descriptorCount = 1;
        samplerBindings.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        // We make use of immutable samplers as they will be shared among all the pipelines and
        // don't change over the lifetime of the pipelines
        samplerBindings.pImmutableSamplers = &m_samplers[s];
      }
    }

    // Prepare constant buffer descriptor
    if(pDesc.hasConstantData)
    {
      VkDescriptorSetLayoutCreateInfo& constantBindingSetInfo = descriptorSetLayoutInfos[m_constantBufferSetIndex];
      if(!constantBindingSetInfo.pBindings)
      {
        // Starting a dedicated set for the constant buffer?
        assert(!constantBindingSetInfo.bindingCount);
        constantBindingSetInfo.pBindings = setBindings[m_constantBufferSetIndex].data();
      }

      VkDescriptorSetLayoutBinding& constantBinding = setBindings[m_constantBufferSetIndex][constantBindingSetInfo.bindingCount++];

      constantBinding.binding            = constantBufferBindingOffset;
      constantBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      constantBinding.descriptorCount    = 1;
      constantBinding.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
      constantBinding.pImmutableSamplers = nullptr;
    }

    // Prepare image/texture descriptors
    VkDescriptorSetLayoutCreateInfo& resourceBindingSetInfo = descriptorSetLayoutInfos[m_resourcesSetIndex];
    if(!resourceBindingSetInfo.pBindings)
    {
      // Starting a dedicated set of descriptors for the images?
      assert(!resourceBindingSetInfo.bindingCount);
      resourceBindingSetInfo.pBindings = setBindings[m_resourcesSetIndex].data();
    }

    for(uint32_t r = 0; r < pDesc.resourceRangesNum; ++r)
    {
      const nrd::ResourceRangeDesc& range = pDesc.resourceRanges[r];

      for(uint32_t b = 0; b < range.descriptorsNum; ++b)
      {
        VkDescriptorSetLayoutBinding& resourceBindings = setBindings[m_resourcesSetIndex][resourceBindingSetInfo.bindingCount++];
        resourceBindings.descriptorCount    = 1;
        resourceBindings.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        resourceBindings.pImmutableSamplers = nullptr;

        switch(range.descriptorType)
        {
          case nrd::DescriptorType::TEXTURE:
            resourceBindings.binding        = resourcesBindingOffset + range.baseRegisterIndex + b;
            resourceBindings.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            break;
          case nrd::DescriptorType::STORAGE_TEXTURE:
            resourceBindings.binding        = storageTextureAndBufferOffset + range.baseRegisterIndex + b;
            resourceBindings.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            break;
          default:
            assert(0);
        }
      }
    }

    // Now lets build the layouts, descriptor sets and pipelines
    NRDPipeline& nrdPipeline = m_pipelines[p];
    nrdPipeline.descriptorLayouts.resize(numPipelineSets, VK_NULL_HANDLE);

    uint32_t numBindings = 0;
    for(uint32_t s = 0; s < numPipelineSets; ++s)
    {
      NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutInfos[s], nullptr, &nrdPipeline.descriptorLayouts[s]));
      numBindings += descriptorSetLayoutInfos[s].bindingCount;
    }

    nrdPipeline.numBindings = numBindings;

    LOGI("Pipeline uses %d bindings\n", numBindings);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr};
    pipelineLayoutInfo.pSetLayouts    = nrdPipeline.descriptorLayouts.data();
    pipelineLayoutInfo.setLayoutCount = numPipelineSets;

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &nrdPipeline.pipelineLayout));

    VkShaderModule computeShaderModule =
        nvvk::createShaderModule(m_device, (const char*)pDesc.computeShaderSPIRV.bytecode, pDesc.computeShaderSPIRV.size / 4);
    assert(computeShaderModule);

    VkPipelineShaderStageCreateInfo stageCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr};
    stageCreateInfo.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCreateInfo.module                          = computeShaderModule;
    stageCreateInfo.pName                           = pDesc.shaderEntryPointName;

    VkComputePipelineCreateInfo pipelineCreateInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr};
    pipelineCreateInfo.layout                      = nrdPipeline.pipelineLayout;
    pipelineCreateInfo.stage                       = stageCreateInfo;

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
  const nrd::LibraryDesc&  lDesc = nrd::GetLibraryDesc();
  const nrd::InstanceDesc& iDesc = nrd::GetInstanceDesc(*m_instance);
  const nrd::PipelineDesc& pDesc = iDesc.pipelines[dispatchDesc.pipelineIndex];

  // These are the base binding indices for each type of textures
  const uint32_t constantBufferBindingOffset   = lDesc.spirvBindingOffsets.constantBufferOffset;
  const uint32_t texturesBindingOffset         = lDesc.spirvBindingOffsets.textureOffset;
  const uint32_t storageTextureAndBufferOffset = lDesc.spirvBindingOffsets.storageTextureAndBufferOffset;
  const uint32_t samplerBindingOffset          = lDesc.spirvBindingOffsets.samplerOffset;

  NRDPipeline& pipeline = m_pipelines[dispatchDesc.pipelineIndex];

  std::vector<VkWriteDescriptorSet>  descriptorUpdates(pipeline.numBindings + iDesc.samplersNum,
                                                       nvvk::make<VkWriteDescriptorSet>());
  std::vector<VkDescriptorImageInfo> descriptorImageInfos(pipeline.numBindings + iDesc.samplersNum);

  std::vector<VkImageMemoryBarrier> imageBarriers;

  auto transitionToShaderRead = [&](VkImage image) {
    VkImageMemoryBarrier barrier = nvvk::makeImageMemoryBarrier(image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    imageBarriers.push_back(barrier);
  };
  auto transitionToShaderWrite = [&](VkImage image) {
    VkImageMemoryBarrier barrier = nvvk::makeImageMemoryBarrier(image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    imageBarriers.push_back(barrier);
  };

  // This piece of code is actually not prepared for having separate sets
  // for each type of descriptor.
  // If needed we would have to have one call to vkCmdPushDescriptorSetKHR for each type.
  assert(m_constantBufferSetIndex == m_resourcesSetIndex && m_samplersSetIndex == m_resourcesSetIndex);

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

      nvvk::Texture* texture = nullptr;
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

      assert(texture);

      // We assume, images bound to storage bindings will be written to, while images bound to
      // texture bindings will be read from.
      // This is a rather simple scheme. If it turns out, these barriers cost too much performance,
      // we might want to be more clever about it by caching transitions between pipelines.
      isStorage ? transitionToShaderWrite(texture->image) : transitionToShaderRead(texture->image);

      VkDescriptorImageInfo& imageInfo = descriptorImageInfos[numResourceUpdates];
      imageInfo.imageView              = texture->descriptor.imageView;
      imageInfo.imageLayout            = VK_IMAGE_LAYOUT_GENERAL;
      ++numResourceUpdates;
    }
  }

  // Issue "dummy" sampler updates to push the immutable samplers
  for(uint32_t s = 0; s < iDesc.samplersNum; ++s)
  {
    // The push descriptor update for immutable samplers will ignore the sampler in 'samplerInfo'
    // and instead push the immutable sampler set up when the pipeline was created.
    VkDescriptorImageInfo& samplerInfo = descriptorImageInfos[numResourceUpdates];
    VkWriteDescriptorSet&  update      = descriptorUpdates[numResourceUpdates];
    update.dstBinding                  = samplerBindingOffset + iDesc.samplersBaseRegisterIndex + s;
    update.descriptorCount             = 1;
    update.descriptorType              = VK_DESCRIPTOR_TYPE_SAMPLER;
    update.pImageInfo                  = &samplerInfo;

    ++numResourceUpdates;
  }

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = m_constantBuffer.buffer;
  bufferInfo.offset = 0;
  bufferInfo.range  = VK_WHOLE_SIZE;

  if(pDesc.hasConstantData)
  {
    VkWriteDescriptorSet constantBufferUpdate(nvvk::make<VkWriteDescriptorSet>());
    constantBufferUpdate.dstBinding      = constantBufferBindingOffset;
    constantBufferUpdate.descriptorCount = 1;
    constantBufferUpdate.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    constantBufferUpdate.pBufferInfo     = &bufferInfo;

    descriptorUpdates[numResourceUpdates++] = constantBufferUpdate;

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
  // Transition all resources into their appropriate state
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, imageBarriers.size(), imageBarriers.data());

  // Update the descriptors. Notice how push descriptors don't require us to make sure the
  // descriptors are not in use anymore.
  vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipelineLayout, m_resourcesSetIndex,
                            numResourceUpdates, descriptorUpdates.data());

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

  // Go!
  vkCmdDispatch(commandBuffer, dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
}
