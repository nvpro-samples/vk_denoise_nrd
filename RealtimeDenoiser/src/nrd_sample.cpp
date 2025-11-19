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

//////////////////////////////////////////////////////////////////////////
/*

 This sample loads GLTF scenes and renders them using RTX (path tracer)

 The path tracer renders into multiple G-Buffers, which are used
 to denoise the image using NRD.
 */
//////////////////////////////////////////////////////////////////////////

// #include <iostream>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_vulkan.h>

#include "nvapp/application.hpp"

#include "nvvk/ray_picker.hpp"
#include "nvvk/sbt_generator.hpp"

#include "nvapp/elem_camera.hpp"
#include "nvapp/elem_dbgprintf.hpp"
#include "nvapp/elem_default_title.hpp"
#include "nvapp/elem_default_menu.hpp"
#include "nvapp/elem_logger.hpp"

#include "nvgui/file_dialog.hpp"
#include "nvgui/property_editor.hpp"
#include "nvgui/sky.hpp"
#include "nvgui/camera.hpp"
#include "nvgui/tonemapper.hpp"

#include "nvnsight/nsightevents.hpp"

#include "nvutils/logger.hpp"
#include "nvutils/file_operations.hpp"
#include "nvutils/camera_manipulator.hpp"

#include "nvvk/barriers.hpp"
#include "nvvk/compute_pipeline.hpp"
#include "nvvk/default_structs.hpp"
#include "nvvk/gbuffers.hpp"
#include "nvvk/context.hpp"
#include "nvvk/descriptors.hpp"
#include "nvvk/shaders.hpp"
#include "nvvk/validation_settings.hpp"

#include "nvvkgltf/scene_rtx.hpp"
#include "nvvkgltf/scene_vk.hpp"

#include "nvvk/hdr_ibl.hpp"
#include "nvvk/pipeline.hpp"

#include "nvshaders_host/sky.hpp"
#include "nvshaders_host/tonemapper.hpp"

#include "nrd.slang.h"
#include "nrd_rchit.slang.h"
#include "nrd_rmiss.slang.h"
#include "pathtrace_rahit.slang.h"
#include "pathtrace_rchit.slang.h"
#include "pathtrace_rmiss.slang.h"
#include "compositing.slang.h"
#include "taa.slang.h"

#include "tonemapper.slang.h"
#include "sky_physical.slang.h"
#include "hdr_dome.slang.h"

#include "shaders/host_device.h"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/sky_io.h.slang"

#include "nrd_wrapper.hpp"
#include "nrd_ui.h"

#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>

#include <array>
#include <filesystem>
#include <math.h>
#include <memory>

using namespace glm;
using namespace nvvk;

template <typename T, size_t N>
constexpr size_t arraySize(T (&)[N])
{
  return N;
}

std::shared_ptr<nvapp::ElementCamera>    g_elem_camera;
std::shared_ptr<nvapp::ElementDbgPrintf> g_dbgPrintf;

// Little desparate helper to allo me set a breakpoint on that exit()
void myExit()
{
  exit(EXIT_FAILURE);
}


// #NRD
// halton low discrepancy sequence, from https://www.shadertoy.com/view/wdXSW8
vec2 halton(int index)
{
  const vec2 coprimes = vec2(2.0F, 3.0F);
  vec2       s        = vec2(index, index);
  vec4       a        = vec4(1, 1, 0, 0);
  while(s.x > 0. && s.y > 0.)
  {
    a.x = a.x / coprimes.x;
    a.y = a.y / coprimes.y;
    a.z += a.x * fmod(s.x, coprimes.x);
    a.w += a.y * fmod(s.y, coprimes.y);
    s.x = floorf(s.x / coprimes.x);
    s.y = floorf(s.y / coprimes.y);
  }
  return vec2(a.z, a.w);
}

// Main sample class
class NrdApplet : public nvapp::IAppElement
{
  enum GbufferNames
  {
    eGBufLdr,
    eGBufDiffRadianceHitDist,     // diffuse radiance and distance to first secondary hit
    eGBufSpecRadianceHitDist,     // specular radiance and distance to
    eGBufOutDiffRadianceHitDist,  // denoised diffuse output
    eGBufOutSpecRadianceHitDist,  // denoised specular output
    eGBufNormalRoughness,         // encoded worldspace normal and linear roughness
    eGBufMotionVectors,           // 3D Object motion (NRD calculates camera motion on its own)
    eGBufViewZ,                   // linear viewspace depth
    eGBufOutDebugView,            // NRD debug view
    eGBufDenoisedUnpacked,        // unpacked denoised result
    eGBufDirectLighting,          // direct lighting
    eGBufTaa,                     // out from TAA

    eGBufNumBuffers
  };

  struct Settings
  {
    int       maxFrames{200000};
    int       maxDepth{5};
    glm::vec4 envIntensity{1.F};
    float     envRotation{0.F};
  } m_settings;

public:
  NrdApplet()           = default;
  ~NrdApplet() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice         = app->getPhysicalDevice();
    allocator_info.device                 = app->getDevice();
    allocator_info.instance               = app->getInstance();
    allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    //FIXME: no way for onAttach to return failure
    NVVK_CHECK(m_alloc.init(allocator_info));  // Allocator


    m_stagingUploader.init(&m_alloc);  // void
    m_stagingUploader.setEnableLayoutBarriers(true);

    m_samplerPool.init(m_device);  // void

    m_sceneVk.init(&m_alloc, &m_samplerPool);  // GLTF Scene buffers
    m_sceneRtx.init(&m_alloc);  //void                                                               // GLTF Scene BLAS/TLAS

    m_tonemapper.init(&m_alloc, tonemapper_slang);  // void
    m_picker.init(&m_alloc);

    m_skyEnv.init(&m_alloc, sky_physical_slang);  //void
    NVVK_CHECK(m_alloc.createBuffer(m_skyParamBuffer, sizeof(shaderio::SkyPhysicalParameters), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));

    m_hdrEnv.init(&m_alloc, &m_samplerPool);  //void


    // Requesting ray tracing properties (this can be moved into m_sbt.init()
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &rt_prop;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create the Shading Binding Table (SBT)
    uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.init(m_app->getDevice(), rt_prop);  // void

    m_viewSize = {app->getWindowSize().width, app->getWindowSize().height};

    NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_bFrameInfo.buffer);

    // Create resources for NRD
    createGbuffers(m_viewSize);

    createCompositionPipeline();
    createTaaPipeline();

    m_cameraManip = std::make_shared<nvutils::CameraManipulator>();
    g_elem_camera->setCameraManipulator(m_cameraManip);
  }

  void createCompositionPipeline()
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBindings(
        {{uint32_t(shaderio::CompositionBindings::eInDiffuse), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {uint32_t(shaderio::CompositionBindings::eInSpecular), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {uint32_t(shaderio::CompositionBindings::eInDirect), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {uint32_t(shaderio::CompositionBindings::eInBaseColor_Metalness), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {uint32_t(shaderio::CompositionBindings::eInNormal_Roughness), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {uint32_t(shaderio::CompositionBindings::eInViewZ), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {uint32_t(shaderio::CompositionBindings::eCompImage), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT}});

    NVVK_CHECK(m_compositionBindings.init(bindings, m_device, 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));
    NVVK_DBG_NAME(m_compositionBindings.getLayout());

    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::RtxPushConstant)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts    = m_compositionBindings.getLayoutPtr();

    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &push_constant;

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_compositionPipelineLayout));

    VkShaderModule assembleShader;

    NVVK_CHECK(nvvk::createShaderModule(assembleShader, m_device, {compositing_slang}));

    VkPipelineShaderStageCreateInfo stageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr};
    stageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCreateInfo.module = assembleShader;
    stageCreateInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr};
    pipelineInfo.layout = m_compositionPipelineLayout;
    pipelineInfo.stage  = stageCreateInfo;


    NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_compositionPipeline));

    NVVK_DBG_NAME(m_compositionPipeline);

    vkDestroyShaderModule(m_device, assembleShader, nullptr);
  }

  void createTaaPipeline()
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBindings(
        {{uint32_t(shaderio::TaaBindings::eInImage), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {uint32_t(shaderio::TaaBindings::eOutImage), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT}});
    m_taaBindings.init(bindings, m_device, 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
    NVVK_DBG_NAME(m_taaBindings.getLayout());

    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(float)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts    = m_taaBindings.getLayoutPtr();

    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &push_constant;

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_taaPipelineLayout));

    VkShaderModule assembleShader;
    NVVK_CHECK(nvvk::createShaderModule(assembleShader, m_device, {taa_slang}));

    VkPipelineShaderStageCreateInfo stageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr};
    stageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCreateInfo.module = assembleShader;
    stageCreateInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr};
    pipelineInfo.layout = m_taaPipelineLayout;
    pipelineInfo.stage  = stageCreateInfo;


    NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_taaPipeline));

    NVVK_DBG_NAME(m_taaPipeline);

    vkDestroyShaderModule(m_device, assembleShader, nullptr);
  }


  void compose(VkCommandBuffer& commandBuffer, VkImageView outImage)
  {
    //FIXME: use descriptorpack instead
    std::vector<VkWriteDescriptorSet> writes;

    VkDescriptorImageInfo outImageInfo = {VK_NULL_HANDLE, outImage, VK_IMAGE_LAYOUT_GENERAL};
    {
      VkWriteDescriptorSet descriptorWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr};
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      descriptorWrite.dstBinding      = uint32_t(shaderio::CompositionBindings::eCompImage);
      descriptorWrite.pImageInfo      = &outImageInfo;

      writes.push_back(descriptorWrite);
    }

    auto bindImage = [&](shaderio::CompositionBindings binding, GbufferNames gbufImage) {
      VkWriteDescriptorSet descriptorWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr};
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      descriptorWrite.dstBinding      = uint32_t(binding);
      descriptorWrite.pImageInfo      = &m_gBuffers.getDescriptorImageInfo(uint32_t(gbufImage));

      writes.emplace_back(descriptorWrite);
    };

    bindImage(shaderio::CompositionBindings::eInDiffuse, eGBufOutDiffRadianceHitDist);
    bindImage(shaderio::CompositionBindings::eInSpecular, eGBufOutSpecRadianceHitDist);
    bindImage(shaderio::CompositionBindings::eInDirect, eGBufDirectLighting);
    bindImage(shaderio::CompositionBindings::eInBaseColor_Metalness, eGBufLdr);
    bindImage(shaderio::CompositionBindings::eInNormal_Roughness, eGBufNormalRoughness);
    bindImage(shaderio::CompositionBindings::eInViewZ, eGBufViewZ);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositionPipeline);

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositionPipelineLayout, 0,
                              (uint32_t)writes.size(), writes.data());
    vkCmdPushConstants(commandBuffer, m_rtPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::RtxPushConstant), &m_pushConst);

    VkExtent2D group_counts = getGroupCounts(m_gBuffers.getSize(), 16);
    vkCmdDispatch(commandBuffer, group_counts.width, group_counts.height, 1);
  }


  void applyTaa(VkCommandBuffer& commandBuffer)
  {
    // FIXME: use descriptorpack instead
    std::vector<VkWriteDescriptorSet> writes;

    auto bindImage = [&](shaderio::TaaBindings binding, GbufferNames gbufImage) {
      VkWriteDescriptorSet descriptorWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr};
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      descriptorWrite.dstBinding      = uint32_t(binding);
      descriptorWrite.pImageInfo      = &m_gBuffers.getDescriptorImageInfo(uint32_t(gbufImage));

      writes.emplace_back(descriptorWrite);
    };

    bindImage(shaderio::TaaBindings::eInImage, eGBufDenoisedUnpacked);
    bindImage(shaderio::TaaBindings::eOutImage, eGBufTaa);

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_taaPipelineLayout, 0,
                              (uint32_t)writes.size(), writes.data());

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_taaPipeline);
    float alpha = 0.1F;
    vkCmdPushConstants(commandBuffer, m_taaPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &alpha);

    VkExtent2D group_counts = getGroupCounts(m_gBuffers.getSize(), 16);
    vkCmdDispatch(commandBuffer, group_counts.width, group_counts.height, 1);
  }


  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    vkDeviceWaitIdle(m_device);

    m_viewSize = {size.width, size.height};
    createGbuffers(m_viewSize);
  }

  void onUIMenu() override
  {
    bool load_file{false};

    windowTitle();

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Load", "Ctrl+O"))
      {
        load_file = true;
      }
      ImGui::Separator();
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      load_file = true;
    }

    if(load_file)
    {
      auto filename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
                                                  "glTF(.gltf, .glb), HDR(.hdr)|*.gltf;*.glb;*.hdr");
      onFileDrop(filename.c_str());
    }
  }

  void onFileDrop(const std::filesystem::path& filename) override
  {
    namespace fs = std::filesystem;

    // Make sure none of the resources is still in use
    vkDeviceWaitIdle(m_device);

    auto extension = filename.extension();
    if(extension == fs::path(".gltf") || extension == fs::path(".glb"))
    {
      createScene(filename);
    }
    else if(extension == ".hdr")
    {
      createHdr(filename);
      resetFrame();
    }

    resetFrame();
  }

  void onUIRender() override
  {
    using namespace nvgui;

    bool reset{false};
    // Pick under mouse cursor
    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
    {
      screenPicking();
    }

    {  // Setting menu
      ImGui::Begin("Settings");

      if(ImGui::CollapsingHeader("Camera"))
      {
        CameraWidget(m_cameraManip);
      }

      if(ImGui::CollapsingHeader("Settings"))
      {
        PropertyEditor::begin();

        if(PropertyEditor::treeNode("Ray Tracing"))
        {
          reset |= PropertyEditor::entry("Depth", [&] { return ImGui::SliderInt("#1", &m_settings.maxDepth, 1, 10); });
          reset |= PropertyEditor::entry("Frames",
                                         [&] { return ImGui::DragInt("#3", &m_settings.maxFrames, 5.0F, 1, 1000000); });
          ImGui::SliderFloat("Override Roughness", &m_pushConst.overrideRoughness, 0, 1, "%.3f");
          ImGui::SliderFloat("Override Metalness", &m_pushConst.overrideMetallic, 0, 1, "%.3f");

          PropertyEditor::treePop();
        }
        bool flipBitangent = m_pushConst.bitangentFlip < 0 ? true : false;
        PropertyEditor::entry("Flip Bitangent", [&] { return ImGui::Checkbox("##5", &flipBitangent); });
        m_pushConst.bitangentFlip = flipBitangent ? -1.0f : 1.0f;

        bool usePSR = !!(m_frameInfo.flags & FLAGS_USE_PSR);
        PropertyEditor::entry("Use PSR", [&] { return ImGui::Checkbox("##6", &usePSR); }, "Use Primary Surface Replacement on mirrors");
        m_frameInfo.flags = (m_frameInfo.flags & ~FLAGS_USE_PSR) | (usePSR ? FLAGS_USE_PSR : 0);


        bool useRegularization = !!(m_frameInfo.flags & FLAGS_USE_PATH_REGULARIZATION);
        PropertyEditor::entry(
            "Use Path Regularization", [&] { return ImGui::Checkbox("##7", &useRegularization); },
            "Use max. roughness propagation to improve indirect specular highlights");
        m_frameInfo.flags = (m_frameInfo.flags & ~FLAGS_USE_PATH_REGULARIZATION)
                            | (useRegularization ? FLAGS_USE_PATH_REGULARIZATION : 0);

        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Environment"))
      {
        int useSky = m_frameInfo.flags & FLAGS_ENVMAP_SKY;
        reset |= ImGui::RadioButton("Sky", &useSky, FLAGS_ENVMAP_SKY);
        ImGui::SameLine();
        reset |= ImGui::RadioButton("Hdr", &useSky, 0);
        m_frameInfo.flags = (m_frameInfo.flags & ~FLAGS_ENVMAP_SKY) | useSky;

        PropertyEditor::begin();
        PropertyEditor::entry(
            "Intensity",
            [&] {
              static float intensity = 1.0f;
              bool hit = ImGui::SliderFloat("##Color", &intensity, 0, 100, "%.3f", ImGuiSliderFlags_Logarithmic);
              m_settings.envIntensity = glm::vec4(intensity, intensity, intensity, 1);
              return hit;
            },
            "HDR multiplier");

        if(!(m_frameInfo.flags & FLAGS_ENVMAP_SKY))
        {
          PropertyEditor::entry("Rotation", [&] { return ImGui::SliderAngle("Rotation", &m_settings.envRotation); }, "Rotating the environment");
        }
        else
        {
          nvgui::skyPhysicalParameterUI(m_skyParams);
        }

        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        nvgui::tonemapperWidget(m_tonemapperData);
      }

      if(ImGui::CollapsingHeader("NRD", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();

        const char* const items[] = {"ReLAX", "ReBLUR", "Reference"};
        if(PropertyEditor::entry("Method", [&]() {
             return ImGui::ListBox("Method", &m_pushConst.method, items, (int)arraySize(items));
           }))
        {
          reset = true;
        }

        PropertyEditor::entry("Split", [&]() { return ImGui::SliderFloat("#Split", &m_splitScreen, 0.0, 1.0f); });

        if(PropertyEditor::entry("Denoiser Values", [&]() { return ImGui::Button("Reset"); }))
        {
          reset            = true;
          m_reblurSettings = nrd::ReblurSettings();
          m_relaxSettings  = nrd::RelaxSettings();
        }
        Nrd_ui::render(m_reblurSettings, m_relaxSettings);

        PropertyEditor::end();
      }

      // #NRD
      if(ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
      {
        ImVec2 tumbnailSize = {100 * m_gBuffers.getAspectRatio(), 100};

        auto showBuffer = [&](const char* name, GbufferNames buffer) {
          ImGui::Text("%s", name);
          if(ImGui::ImageButton(name, (ImTextureID)m_gBuffers.getDescriptorSet(buffer), tumbnailSize))
            m_showBuffer = buffer;
        };

        if(ImGui::BeginTable("thumbnails", 2))
        {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          showBuffer("Diffuse Radiance\n(REBLUR: YCoCg)", eGBufDiffRadianceHitDist);
          ImGui::TableNextColumn();
          showBuffer("Specular Radiance\n(REBLUR: YCoCg)", eGBufSpecRadianceHitDist);
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          showBuffer("Normal/Roughness", eGBufNormalRoughness);
          ImGui::TableNextColumn();
          showBuffer("Denoised", eGBufDenoisedUnpacked);
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          showBuffer("TAA", eGBufTaa);
          ImGui::TableNextColumn();
          showBuffer("LDR", eGBufLdr);
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          showBuffer("NRD Debug", eGBufOutDebugView);

          ImGui::EndTable();
        }
      }

      ImGui::End();

      if(reset)
      {
        resetFrame();
      }
    }

    {
      // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(m_showBuffer), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NXPROFILEFUNC("onRender");

    if(!m_scene.valid())
    {
      return;
    }

    NVVK_DBG_SCOPE(cmd);

    // Get camera info
    float view_aspect_ratio = (float)m_viewSize.x / m_viewSize.y;

    m_frameInfo.prevMVP = m_frameInfo.proj * m_frameInfo.view;

    // Update Frame buffer uniform buffer
    const auto& clip = m_cameraManip->getClipPlanes();
    m_frameInfo.view = m_cameraManip->getViewMatrix();
    m_frameInfo.proj = glm::perspectiveRH_ZO(glm::radians(m_cameraManip->getFov()), view_aspect_ratio, clip.x, clip.y);

    auto unflippedProj = m_frameInfo.proj;  // There's some weirness going on with the vertical

    // Were're feeding the raytracer with a flipped matrix for convenience
    m_frameInfo.proj[1][1] *= -1;

    m_frameInfo.projInv      = glm::inverse(m_frameInfo.proj);
    m_frameInfo.viewInv      = glm::inverse(m_frameInfo.view);
    m_frameInfo.envRotation  = m_settings.envRotation;
    m_frameInfo.envIntensity = m_settings.envIntensity;
    m_frameInfo.jitter       = halton(m_frame) - vec2(0.5);

    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &m_frameInfo);

    // Push constant
    m_pushConst.maxDepth   = m_settings.maxDepth;
    m_pushConst.frame      = m_frame;
    m_pushConst.mouseCoord = g_dbgPrintf->getMouseCoord();

    // Helper lambdas to make writing image pipeline barriers easier
    auto imageShaderWriteToRead = [](VkImage image, VkPipelineStageFlagBits2 srcStage, VkPipelineStageFlagBits2 dstStage) {
      return nvvk::makeImageMemoryBarrier({
          .image         = image,
          .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
          .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
          .srcStageMask  = srcStage,
          .dstStageMask  = dstStage,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      });
    };
    auto imageShaderReadToWrite = [](VkImage image, VkPipelineStageFlagBits2 srcStage, VkPipelineStageFlagBits2 dstStage) {
      return nvvk::makeImageMemoryBarrier({.image         = image,
                                           .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
                                           .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
                                           .srcStageMask  = srcStage,
                                           .dstStageMask  = dstStage,
                                           .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                           .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT});
    };

    auto gbufferShaderWriteToRead = [&]<typename T, size_t N, typename G>(const G& gbuffer, const T(&buffers)[N],
                                                                          VkPipelineStageFlagBits2 srcStage,
                                                                          VkPipelineStageFlagBits2 dstStage) {
      std::array<VkImageMemoryBarrier2, N> x;
      for(size_t i = 0; i < N; ++i)
        x[i] = imageShaderWriteToRead(gbuffer.getColorImage(buffers[i]), srcStage, dstStage);
      return x;
    };
    auto gbufferShaderReadToWrite = [&]<typename T, size_t N, typename G>(const G& gbuffer, const T(&buffers)[N],
                                                                          VkPipelineStageFlagBits2 srcStage,
                                                                          VkPipelineStageFlagBits2 dstStage) {
      std::array<VkImageMemoryBarrier2, N> x;
      for(size_t i = 0; i < N; ++i)
        x[i] = imageShaderReadToWrite(gbuffer.getColorImage(buffers[i]), srcStage, dstStage);
      return x;
    };

    auto gBufferShaderWriteToRead = [&]<std::size_t N>(const GbufferNames(&buffers)[N], VkPipelineStageFlagBits2 srcStage,
                                                       VkPipelineStageFlagBits2 dstStage) {
      return gbufferShaderWriteToRead(m_gBuffers, buffers, srcStage, dstStage);
    };
    auto gBufferShaderReadToWrite = [&]<std::size_t N>(const GbufferNames(&buffers)[N], VkPipelineStageFlagBits2 srcStage,
                                                       VkPipelineStageFlagBits2 dstStage) {
      return gbufferShaderReadToWrite(m_gBuffers, buffers, srcStage, dstStage);
    };

    auto cmdImageBarriers = [&](const std::initializer_list<const std::span<const VkImageMemoryBarrier2>>& barriers) {
      std::vector<VkImageMemoryBarrier2> final;
      for(auto b : barriers)
        final.insert(final.end(), b.begin(), b.end());

      const VkDependencyInfo depInfo{.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                     .imageMemoryBarrierCount = (uint32_t) final.size(),
                                     .pImageMemoryBarriers    = final.data()};
      vkCmdPipelineBarrier2(cmd, &depInfo);
    };

    // Make G-Buffers writeable to raytracer
    cmdImageBarriers({gBufferShaderReadToWrite({eGBufDiffRadianceHitDist, eGBufSpecRadianceHitDist, eGBufNormalRoughness,
                                                eGBufMotionVectors, eGBufViewZ, eGBufDirectLighting, eGBufLdr},
                                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)});

    // Pathtrace the scene
    raytraceScene(cmd);

    // Make G-Buffers readable to NRD
    cmdImageBarriers({gBufferShaderWriteToRead({eGBufDiffRadianceHitDist, eGBufSpecRadianceHitDist, eGBufNormalRoughness,
                                                eGBufMotionVectors, eGBufViewZ, eGBufDirectLighting, eGBufLdr},
                                               VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)});

    // #NRD Denoising
    if(m_nrd)
    {
      // Set NRD settings
      {
        // Update per-Frame settings
        memcpy(m_nrdSettings.viewToClipMatrixPrev, m_nrdSettings.viewToClipMatrix, sizeof(nrd::CommonSettings::viewToClipMatrixPrev));
        memcpy(m_nrdSettings.viewToClipMatrix, glm::value_ptr(unflippedProj), sizeof(nrd::CommonSettings::viewToClipMatrix));
        memcpy(m_nrdSettings.worldToViewMatrixPrev, m_nrdSettings.worldToViewMatrix,
               sizeof(nrd::CommonSettings::worldToViewMatrixPrev));
        memcpy(m_nrdSettings.worldToViewMatrix, glm::value_ptr(m_frameInfo.view), sizeof(nrd::CommonSettings::worldToViewMatrix));

        memcpy(m_nrdSettings.cameraJitterPrev, m_nrdSettings.cameraJitter, sizeof(nrd::CommonSettings::cameraJitterPrev));
        m_nrdSettings.cameraJitter[0] = m_frameInfo.jitter.x;
        m_nrdSettings.cameraJitter[1] = m_frameInfo.jitter.y;

        m_nrdSettings.frameIndex = m_frame;
        m_nrdSettings.accumulationMode =
            (m_frame == 0 ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE);

        m_nrdSettings.resourceSizePrev[0] = m_viewSize[0];
        m_nrdSettings.resourceSizePrev[1] = m_viewSize[1];

        m_nrdSettings.resourceSize[0] = m_viewSize[0];
        m_nrdSettings.resourceSize[1] = m_viewSize[1];

        m_nrdSettings.rectSizePrev[0] = m_viewSize[0];
        m_nrdSettings.rectSizePrev[1] = m_viewSize[1];

        m_nrdSettings.rectSize[0] = m_viewSize[0];
        m_nrdSettings.rectSize[1] = m_viewSize[1];

        // Debug: we don't provide true motions vectors yet
        m_nrdSettings.motionVectorScale[0] = m_nrdSettings.motionVectorScale[1] = 1.0f;
        m_nrdSettings.motionVectorScale[2]                                      = 0.0f;

        m_nrdSettings.isMotionVectorInWorldSpace = true;

        m_nrdSettings.isBaseColorMetalnessAvailable = true;

        // We want to visualize the denoiser's debug texture
        m_nrdSettings.enableValidation = true;

        m_nrd->setCommonSettings(m_nrdSettings);
      }

      switch(m_pushConst.method)
      {
        case NRD_REBLUR: {
          m_nrd->setREBLURSettings(m_reblurSettings);

          nrd::Identifier denoiser = nrd::Identifier(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR);
          // Perform the denoising!
          m_nrd->denoise(&denoiser, 1, cmd);
          break;
        }
        case NRD_RELAX: {
          m_nrd->setRELAXSettings(m_relaxSettings);
          nrd::Identifier denoiser = nrd::Identifier(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR);
          // Perform the denoising!
          m_nrd->denoise(&denoiser, 1, cmd);
          break;
        }
        default: {
          auto poolTextureFromGBufTexture = [&](GbufferNames gbufIndex) -> nvvk::Image {
            return {.image = m_gBuffers.getColorImage(gbufIndex), .descriptor = {m_gBuffers.getDescriptorImageInfo(gbufIndex)}};
          };
          nrd::Identifier denoisers[] = {nrd::Identifier(nrd::Denoiser::REFERENCE), nrd::Identifier(nrd::Denoiser::REFERENCE) + 1};
          m_nrd->setUserPoolTexture(nrd::ResourceType::IN_SIGNAL, poolTextureFromGBufTexture(eGBufDiffRadianceHitDist));
          m_nrd->setUserPoolTexture(nrd::ResourceType::OUT_SIGNAL, poolTextureFromGBufTexture(eGBufOutDiffRadianceHitDist));
          m_nrd->denoise(&denoisers[0], 1, cmd);
          m_nrd->setUserPoolTexture(nrd::ResourceType::IN_SIGNAL, poolTextureFromGBufTexture(eGBufSpecRadianceHitDist));
          m_nrd->setUserPoolTexture(nrd::ResourceType::OUT_SIGNAL, poolTextureFromGBufTexture(eGBufOutSpecRadianceHitDist));
          m_nrd->denoise(&denoisers[1], 1, cmd);
        }
      }
    }

    // Make denoised outputs readable and denoisedUnpacked writable for composition
    cmdImageBarriers({gBufferShaderWriteToRead({eGBufOutDiffRadianceHitDist, eGBufOutSpecRadianceHitDist},
                                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
                      gBufferShaderReadToWrite({eGBufDenoisedUnpacked}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)});

    // Compose denoised diffuse, specular, and direct lighting
    compose(cmd, m_gBuffers.getColorImageView(eGBufDenoisedUnpacked));

    // Make denoised unpacked readable and TAA buffer writable
    cmdImageBarriers(
        {gBufferShaderWriteToRead({eGBufDenoisedUnpacked}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
         gBufferShaderReadToWrite({eGBufTaa}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)});

    // Apply temporal anti-aliasing
    applyTaa(cmd);

    // Make TAA output readable and LDR writable for tonemapper
    cmdImageBarriers(
        {gBufferShaderWriteToRead({eGBufTaa}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
         gBufferShaderReadToWrite({eGBufLdr}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)});

    // Apply tonemapper (using TAA output)
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eGBufTaa),
                            m_gBuffers.getDescriptorImageInfo(eGBufLdr));

    // Make tonemapped image readable to ImGUI
    cmdImageBarriers({gBufferShaderWriteToRead({eGBufLdr}, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)});

    m_frame++;
  }

private:
  void createScene(const std::filesystem::path& filename)
  {
    m_sceneRtx.destroy();
    m_sceneVk.destroy();
    m_scene.destroy();

    if(!m_scene.load(filename))
    {
      LOGE("Error loading scene");
      return;
    }

    m_cameraManip->fit(m_scene.getSceneBounds().min(), m_scene.getSceneBounds().max());  // Navigation help

    auto cmd = m_app->createTempCmdBuffer();

    {  // Create the Vulkan side of the scene
      m_sceneVk.create(cmd, m_stagingUploader, m_scene);
      m_stagingUploader.cmdUploadAppended(cmd);  //make sure the scene buffers are on the GPU by the time we build
                                                 //the Acceleration Structures
      m_sceneRtx.create(cmd, m_stagingUploader, m_scene, m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);  // Create BLAS / TLAS
      m_stagingUploader.cmdUploadAppended(cmd);
    }

    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Descriptor Set and Pipelines
    createSceneSet();
    createRtxSet();
    createRtxPipeline();  // must recreate due to texture changes
    writeSceneSet();
    writeRtxSet();
  }

  void createGbuffers(const glm::uvec2& size)
  {
    m_viewSize = size;
    VkExtent2D vk_size{static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)};

    std::vector<VkFormat> color_buffers(eGBufNumBuffers);
    color_buffers[eGBufLdr] = VK_FORMAT_R8G8B8A8_UNORM;
    color_buffers[eGBufTaa] = VK_FORMAT_R16G16B16A16_SFLOAT;

    // #NRD Create buffers according to NRD's requirements. Consult NRDDescs.h to learn
    // which (minimum) format is required for each input buffer type.
    color_buffers[eGBufDiffRadianceHitDist]    = VK_FORMAT_R16G16B16A16_SFLOAT;
    color_buffers[eGBufSpecRadianceHitDist]    = VK_FORMAT_R16G16B16A16_SFLOAT;
    color_buffers[eGBufNormalRoughness]        = NRDWrapper::getNormalRoughnessFormat();
    color_buffers[eGBufMotionVectors]          = VK_FORMAT_R16G16B16A16_SFLOAT;
    color_buffers[eGBufViewZ]                  = VK_FORMAT_R16_SFLOAT;
    color_buffers[eGBufOutDiffRadianceHitDist] = VK_FORMAT_R16G16B16A16_SFLOAT;
    color_buffers[eGBufOutSpecRadianceHitDist] = VK_FORMAT_R16G16B16A16_SFLOAT;
    color_buffers[eGBufOutDebugView]           = VK_FORMAT_R8G8B8A8_UNORM;
    color_buffers[eGBufDenoisedUnpacked]       = VK_FORMAT_R16G16B16A16_SFLOAT;
    color_buffers[eGBufDirectLighting]         = VK_FORMAT_R16G16B16A16_SFLOAT;

    // Creation of the GBuffers
    m_gBuffers.deinit();

    VkSamplerCreateInfo createInfo = DEFAULT_VkSamplerCreateInfo;
    createInfo.addressModeU = createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler;
    m_samplerPool.acquireSampler(sampler, createInfo);

    nvvk::GBufferInitInfo gbInfo = {.allocator      = &m_alloc,
                                    .colorFormats   = color_buffers,
                                    .imageSampler   = sampler,
                                    .descriptorPool = m_app->getTextureDescriptorPool()};

    m_gBuffers.init(gbInfo);

    auto cmd = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_gBuffers.update(cmd, vk_size));
    m_app->submitAndWaitTempCmdBuffer(cmd);

    initializeNRD();

    resetFrame();
  }

  void initializeNRD()
  {
    // Create a texture pool for NRD
    std::array<nvvk::Image, size_t(nrd::ResourceType::MAX_NUM)> nrdTexturePool{};

    // Map our G-buffers to NRD resource types
    auto poolTextureFromGBufTexture = [&](GbufferNames gbufIndex) -> nvvk::Image {
      return {.image = m_gBuffers.getColorImage(gbufIndex), .descriptor = m_gBuffers.getDescriptorImageInfo(gbufIndex)};
    };

    nrdTexturePool[size_t(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST)] = poolTextureFromGBufTexture(eGBufDiffRadianceHitDist);
    nrdTexturePool[size_t(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST)] = poolTextureFromGBufTexture(eGBufSpecRadianceHitDist);
    nrdTexturePool[size_t(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST)] = poolTextureFromGBufTexture(eGBufOutDiffRadianceHitDist);
    nrdTexturePool[size_t(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST)] = poolTextureFromGBufTexture(eGBufOutSpecRadianceHitDist);
    nrdTexturePool[size_t(nrd::ResourceType::IN_NORMAL_ROUGHNESS)] = poolTextureFromGBufTexture(eGBufNormalRoughness);
    nrdTexturePool[size_t(nrd::ResourceType::IN_VIEWZ)]            = poolTextureFromGBufTexture(eGBufViewZ);
    nrdTexturePool[size_t(nrd::ResourceType::IN_BASECOLOR_METALNESS)] = poolTextureFromGBufTexture(eGBufLdr);
    nrdTexturePool[size_t(nrd::ResourceType::OUT_VALIDATION)]         = poolTextureFromGBufTexture(eGBufOutDebugView);
    nrdTexturePool[size_t(nrd::ResourceType::IN_MV)]                  = poolTextureFromGBufTexture(eGBufMotionVectors);

    m_nrd.reset();
    m_nrd = std::make_unique<NRDWrapper>(m_alloc, m_app->getQueue(0), m_samplerPool, uint16_t(m_viewSize.x),
                                         uint16_t(m_viewSize.y), nrdTexturePool.data());

    createNrdSet();
    writeNrdSet();
  }

  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
    m_rtPipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
    m_rtPipelineLayout = VK_NULL_HANDLE;
    m_alloc.destroyBuffer(m_sbtBuffer);

    // Creating all shaders
    enum StageIndices
    {
      ePrimaryRaygen,
      ePrimaryClosestHit,
      ePrimaryMiss,
      eSecondaryMiss,
      eSecondaryClosestHit,
      eSecondaryAnyHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point

    // #Raygen
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, {nrd_slang}));
    stage.stage            = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[ePrimaryRaygen] = stage;

    // Miss
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, {pathtrace_rmiss_slang}));
    stage.stage            = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eSecondaryMiss] = stage;

    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, {nrd_rmiss_slang}));
    stage.stage          = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[ePrimaryMiss] = stage;

    // AnyHit
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, {pathtrace_rahit_slang}));
    stage.stage              = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eSecondaryAnyHit] = stage;

    // Hit Group - Closest Hit
    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, {pathtrace_rchit_slang}));
    stage.stage                  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eSecondaryClosestHit] = stage;

    NVVK_CHECK(nvvk::createShaderModule(stage.module, m_device, {nrd_rchit_slang}));
    stage.stage                = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[ePrimaryClosestHit] = stage;

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = ePrimaryRaygen;
    shaderGroups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = ePrimaryMiss;
    shaderGroups.push_back(group);
    group.generalShader = eSecondaryMiss;
    shaderGroups.push_back(group);

    // Primary closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = ePrimaryClosestHit;
    group.anyHitShader     = eSecondaryAnyHit;
    shaderGroups.push_back(group);

    // Secondary closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eSecondaryClosestHit;
    group.anyHitShader     = eSecondaryAnyHit;
    shaderGroups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::RtxPushConstant)};

    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_rtPipelineLayout,
                                          {m_rtBindings.getLayout(), m_sceneBindings.getLayout(),
                                           m_nrdBindings.getLayout(), m_hdrEnv.getDescriptorSetLayout()},
                                          {push_constant}));
    NVVK_DBG_NAME(m_rtPipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    ray_pipeline_info.pGroups                      = shaderGroups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 2;  // Ray depth
    ray_pipeline_info.layout                       = m_rtPipelineLayout;

    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &m_rtPipeline);
    NVVK_DBG_NAME(m_rtPipeline);

    // Creating the SBT
    auto sbtSize = m_sbt.calculateSBTBufferSize(m_rtPipeline, ray_pipeline_info);
    m_alloc.createBuffer(m_sbtBuffer, sbtSize, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
                         VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                         m_sbt.getBufferAlignment());
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    m_sbt.populateSBTBuffer(m_sbtBuffer.address, sbtSize, m_sbtBuffer.mapping);

    // Removing temp modules
    for(auto& s : stages)
    {
      vkDestroyShaderModule(m_device, s.module, nullptr);
    }
  }

  void createNrdSet()
  {
    using shaderio::NrdBindings;

    m_nrdBindings.deinit();

    nvvk::DescriptorBindings d;
    // #NRD
    d.addBindings({{NrdBindings::eNormal_Roughness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL},
                   {NrdBindings::eNoisyDiffuse, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL},
                   {NrdBindings::eNoisySpecular, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL},
                   {NrdBindings::eViewZ, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL},
                   {NrdBindings::eDirectLighting, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL},
                   {NrdBindings::eMotionVectors, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL},
                   {NrdBindings::eBaseColor_Metalness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL}});

    NVVK_CHECK(m_nrdBindings.init(d, m_device));
  }

  void writeNrdSet()
  {
    using shaderio::NrdBindings;

    if(!m_scene.valid())
    {
      return;
    }

    nvvk::WriteSetContainer                           writes;
    std::vector<std::pair<NrdBindings, GbufferNames>> nrdBindings = {
        {NrdBindings::eNormal_Roughness, eGBufNormalRoughness},  {NrdBindings::eNoisyDiffuse, eGBufDiffRadianceHitDist},
        {NrdBindings::eNoisySpecular, eGBufSpecRadianceHitDist}, {NrdBindings::eViewZ, eGBufViewZ},
        {NrdBindings::eDirectLighting, eGBufDirectLighting},     {NrdBindings::eMotionVectors, eGBufMotionVectors},
        {NrdBindings::eBaseColor_Metalness, eGBufLdr},
    };

    for(const auto& [nrdBinding, gbufferName] : nrdBindings)
    {
      writes.append(m_nrdBindings.makeWrite(nrdBinding), &m_gBuffers.getDescriptorImageInfo(gbufferName));
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }


  void createRtxSet()
  {
    m_rtBindings.deinit();

    nvvk::DescriptorBindings d;

    // This descriptor set, holds the top level acceleration structure and the output image
    d.addBinding(shaderio::RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_rtBindings.init(d, m_device));
    NVVK_DBG_NAME(m_rtBindings.getLayout());
  }

  void writeRtxSet()
  {
    if(!m_scene.valid())
    {
      return;
    }

    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_sceneRtx.tlas();

    nvvk::WriteSetContainer writes;
    writes.append(m_rtBindings.makeWrite(shaderio::RtxBindings::eTlas), tlas);

    vkUpdateDescriptorSets(m_device, writes.size(), writes.data(), 0, nullptr);
  }


  void createSceneSet()
  {
    m_sceneBindings.deinit();

    nvvk::DescriptorBindings d;

    // This descriptor set, holds the top level acceleration structure and the output image
    d.addBinding(shaderio::SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk.nbTextures(),
                 VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_sceneBindings.init(d, m_device));
    NVVK_DBG_NAME(m_sceneBindings.getLayout());
  }

  void writeSceneSet()
  {
    if(!m_scene.valid())
    {
      return;
    }

    nvvk::WriteSetContainer writes;

    std::vector<VkDescriptorImageInfo> diit;
    for(const auto& texture : m_sceneVk.textures())  // All texture samplers
    {
      diit.emplace_back(texture.descriptor);
    }
    writes.append(m_sceneBindings.makeWrite(shaderio::SceneBindings::eTextures), diit.data());

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame() { m_frame = 0; }

  void windowTitle()
  {
    // Window Title
    static float dirty_timer = 0.0F;
    dirty_timer += ImGui::GetIO().DeltaTime;
    if(dirty_timer > 1.0F)  // Refresh every seconds
    {
      const auto&           size = m_app->getViewportSize();
      std::array<char, 256> buf{};
      snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms | Frame %d", TARGET_NAME,
               static_cast<int>(size.width), static_cast<int>(size.height), static_cast<int>(ImGui::GetIO().Framerate),
               1000.F / ImGui::GetIO().Framerate, m_frame);
      glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
      dirty_timer = 0;
    }
  }


  //--------------------------------------------------------------------------------------------------
  // Send a ray under mouse coordinates, and retrieve the information
  // - Set new camera interest point on hit position
  //
  void screenPicking()
  {
    auto* tlas = m_sceneRtx.tlas();
    if(tlas == VK_NULL_HANDLE)
      return;

    ImGui::Begin("Viewport");  // ImGui, picking within "viewport"
    auto  mouse_pos        = ImGui::GetMousePos();
    auto  main_size        = ImGui::GetContentRegionAvail();
    auto  corner           = ImGui::GetCursorScreenPos();  // Corner of the viewport
    float aspect_ratio     = main_size.x / main_size.y;
    mouse_pos              = mouse_pos - corner;
    ImVec2 local_mouse_pos = mouse_pos / main_size;
    ImGui::End();

    auto* cmd = m_app->createTempCmdBuffer();

    // Finding current camera matrices
    const auto& view = m_cameraManip->getViewMatrix();
    auto        proj = glm::perspectiveRH_ZO(glm::radians(m_cameraManip->getFov()), aspect_ratio, 0.1F, 1000.0F);
    proj[1][1] *= -1;

    // Setting up the data to do picking
    nvvk::RayPicker::PickInfo pick_info;
    pick_info.pickPos        = {local_mouse_pos.x, local_mouse_pos.y};
    pick_info.modelViewInv   = glm::inverse(view);
    pick_info.perspectiveInv = glm::inverse(proj);
    pick_info.tlas           = m_sceneRtx.tlas();

    // Run and wait for result
    m_picker.run(cmd, pick_info);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Retrieving picking information
    nvvk::RayPicker::PickResult pr = m_picker.getResult();
    if(pr.instanceID == ~0)
    {
      LOGI("Nothing Hit\n");
      return;
    }

    if(pr.hitT <= 0.F)
    {
      LOGI("Hit Distance == 0.0\n");
      return;
    }

    // Find where the hit point is and set the interest position
    glm::vec3 world_pos = glm::vec3(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    m_cameraManip->getLookat(eye, center, up);
    m_cameraManip->setLookat(eye, world_pos, up, false);

    // Logging picking info.
    const nvvkgltf::RenderNode& renderNode = m_scene.getRenderNodes()[pr.instanceID];
    const tinygltf::Node&       node       = m_scene.getModel().nodes[renderNode.refNodeID];

    LOGI("Node Name: %s\n", node.name.c_str());
    LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pr.primitiveID);
    LOGI(" - Render: GltfRenderNode: %d, RenderPrim: %d\n", pr.instanceID, pr.instanceCustomIndex);
    LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
  }

  void raytraceScene(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);

    if(m_sceneVk.sceneDesc().address == 0 || m_sceneVk.sceneDesc().buffer == VK_NULL_HANDLE)
    {
      LOGE("ERROR: Scene descriptor buffer is not initialized!\n");
      return;
    }

    vkCmdUpdateBuffer(cmd, m_skyParamBuffer.buffer, 0, sizeof(m_skyParams), &m_skyParams);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtBindings.getSet(0), m_sceneBindings.getSet(0), m_nrdBindings.getSet(0),
                                           m_hdrEnv.getDescriptorSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);


    m_pushConst.frameInfo = (shaderio::FrameInfo*)m_bFrameInfo.address;
    m_pushConst.gltfScene = (shaderio::GltfScene*)m_sceneVk.sceneDesc().address;
    m_pushConst.skyParams = (shaderio::SkyPhysicalParameters*)m_skyParamBuffer.address;
    vkCmdPushConstants(cmd, m_rtPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::RtxPushConstant), &m_pushConst);

    const auto& size = m_gBuffers.getSize();

    const auto& sbtRegions = m_sbt.getSBTRegions(0);
    vkCmdTraceRaysKHR(cmd, &sbtRegions.raygen, &sbtRegions.miss, &sbtRegions.hit, &sbtRegions.callable, size.width, size.height, 1);
  }

  void createHdr(const std::filesystem::path& filename)
  {
    auto cmd = m_app->createTempCmdBuffer();
    m_hdrEnv.destroyEnvironment();
    m_hdrEnv.loadEnvironment(cmd, m_stagingUploader, filename);
    m_stagingUploader.cmdUploadAppended(cmd);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_stagingUploader.releaseStaging();
  }

  void destroyResources()
  {
    m_nrd.reset();

    m_alloc.destroyBuffer(m_bFrameInfo);

    m_sceneRtx.deinit();
    m_sceneVk.deinit();
    m_scene.destroy();

    m_hdrEnv.deinit();
    m_skyEnv.deinit();
    m_alloc.destroyBuffer(m_skyParamBuffer);

    m_gBuffers.deinit();

    vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
    m_rtPipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
    m_rtPipelineLayout = VK_NULL_HANDLE;

    m_nrdBindings.deinit();
    m_rtBindings.deinit();
    m_sceneBindings.deinit();

    vkDestroyPipeline(m_device, m_compositionPipeline, nullptr);
    m_compositionBindings.deinit();
    vkDestroyPipelineLayout(m_device, m_compositionPipelineLayout, nullptr);
    m_compositionPipelineLayout = VK_NULL_HANDLE;

    vkDestroyPipeline(m_device, m_taaPipeline, nullptr);
    m_taaBindings.deinit();
    vkDestroyPipelineLayout(m_device, m_taaPipelineLayout, nullptr);
    m_taaPipelineLayout = VK_NULL_HANDLE;


    m_alloc.destroyBuffer(m_sbtBuffer);
    m_sbt.deinit();

    m_picker.deinit();
    m_tonemapper.deinit();
    m_samplerPool.deinit();

    m_stagingUploader.deinit();
    m_alloc.deinit();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  VkDevice m_device = VK_NULL_HANDLE;

  nvapp::Application*     m_app{nullptr};
  nvvk::ResourceAllocator m_alloc{};  // The VMA allocator
  nvvk::StagingUploader   m_stagingUploader{};

  glm::uvec2 m_viewSize = {1, 1};

  // #NRD
  nvvk::GBuffer        m_gBuffers;     // G-Buffers: color + depth
  nvvk::DescriptorPack m_nrdBindings;  // NRD descriptor set

  // #NRD
  std::unique_ptr<NRDWrapper> m_nrd;
  nrd::CommonSettings         m_nrdSettings    = {};
  nrd::RelaxSettings          m_relaxSettings  = {};
  nrd::ReblurSettings         m_reblurSettings = {};

  // Additional UI state
  float        m_splitScreen = 0.0f;
  GbufferNames m_showBuffer  = eGBufLdr;

  // Resources
  nvvk::Buffer m_bFrameInfo;

  // Pipeline
  shaderio::RtxPushConstant m_pushConst{
      .frame                  = -1,
      .maxLuminance           = 10000.f,
      .maxDepth               = 7,
      .method                 = NRD_REBLUR,
      .meterToUnitsMultiplier = 1.0,
      .overrideRoughness      = -1.0,
      .overrideMetallic       = -1.0,
      .mouseCoord             = {0, 0},
      .bitangentFlip          = 1.0,
  };  // Information sent to the shader

  int m_frame{0};

  nvvk::DescriptorPack m_sceneBindings;  // Scene texture descriptors

  nvvk::DescriptorPack    m_rtBindings{};
  nvvk::WriteSetContainer m_rtWriteSetContainer{};

  VkPipelineLayout m_rtPipelineLayout{};
  VkPipeline       m_rtPipeline{};

  //FIXME: there is no reason that we must pass m_cameraManip around as a shared_ptr excepto for the CameraWidget wills it so.
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip;

  shaderio::FrameInfo m_frameInfo{.flags = FLAGS_USE_PSR | FLAGS_USE_PATH_REGULARIZATION};

  nvvkgltf::Scene    m_scene;
  nvvkgltf::SceneVk  m_sceneVk;
  nvvkgltf::SceneRtx m_sceneRtx;

  nvvk::SBTGenerator m_sbt;  // Shading binding table wrapper
  nvvk::Buffer       m_sbtBuffer;

  nvvk::RayPicker   m_picker;  // For ray picking info
  nvvk::HdrIbl      m_hdrEnv;
  nvvk::SamplerPool m_samplerPool;  // HdrEnvDome wants this

  nvshaders::SkyPhysical          m_skyEnv;
  shaderio::SkyPhysicalParameters m_skyParams;
  nvvk::Buffer                    m_skyParamBuffer;

  nvshaders::Tonemapper    m_tonemapper;
  shaderio::TonemapperData m_tonemapperData = {};

  // Assemble compute shader
  VkPipeline           m_compositionPipeline = {};
  nvvk::DescriptorPack m_compositionBindings;
  VkPipelineLayout     m_compositionPipelineLayout = {};
  nvvk::DescriptorPack m_taaBindings;
  VkPipeline           m_taaPipeline       = {};
  VkPipelineLayout     m_taaPipelineLayout = {};
};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

int main(int, char**)
{
  nvapp::ApplicationCreateInfo appInitInfo;
  appInitInfo.name  = TARGET_NAME " Example";
  appInitInfo.vSync = true;
  // spec.headless = true;
  // spec.headlessFrameCount = 10;

  if(appInitInfo.headless)
  {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
  }

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR    ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};

  nvvk::ContextInitInfo ctxInfo{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},

      .deviceExtensions = {{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME},
                           {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature},
                           {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature},
                           {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                           {VK_KHR_RAY_QUERY_EXTENSION_NAME, &ray_query_features, appInitInfo.headless == true},
                           {VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeature},
                           {VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME},
                           {VK_KHR_SWAPCHAIN_EXTENSION_NAME},
                           {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeature},
                           {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
  };

#if NVVK_SUPPORTS_AFTERMATH
  // Optional extension to support Aftermath shader level debugging
  ctxInfo.deviceExtension.emplace_back({VK_KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_EXTENSION_NAME, true});
#endif

  nvvk::addSurfaceExtensions(ctxInfo.instanceExtensions);

  nvvk::ValidationSettings validation{};
  {
    // Enable Debug stuff
    validation.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);

    // Danger: keep validation alive until after vkCtx.init()
    ctxInfo.instanceCreateInfoExt = validation.buildPNextChain();

    g_dbgPrintf = std::make_shared<nvapp::ElementDbgPrintf>();
  }

  // We need one queue. This queue will have "queue family index 0"
  ctxInfo.queues = {VK_QUEUE_GRAPHICS_BIT};

  nvvk::Context vkCtx;
  if(vkCtx.init(ctxInfo) != VK_SUCCESS)
  {
    return EXIT_FAILURE;
  }

  appInitInfo.instance       = vkCtx.getInstance();
  appInitInfo.physicalDevice = vkCtx.getPhysicalDevice();
  appInitInfo.device         = vkCtx.getDevice();
  appInitInfo.queues.push_back(vkCtx.getQueueInfo(0));

  // Create the application
  nvapp::Application app;
  app.init(appInitInfo);

  // Create application elements
  std::shared_ptr<nvapp::IAppElement> dlss_applet = std::make_shared<NrdApplet>();
  g_elem_camera                                   = std::make_shared<nvapp::ElementCamera>();

  app.addElement(g_elem_camera);
  app.addElement(dlss_applet);
  app.addElement(g_dbgPrintf);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // Menu / Quit

  // Search paths
  std::vector<std::filesystem::path> default_search_paths = {
      ".", "..", "../..", "../../..", nvutils::getExecutablePath().parent_path() / TARGET_EXE_TO_DOWNLOAD_DIRECTORY};

  // Load HDR
  std::filesystem::path hdr_file = nvutils::findFile(R"(environment.hdr)", default_search_paths);
  dlss_applet->onFileDrop(hdr_file);

  // Load scene
  std::filesystem::path scn_file = nvutils::findFile(R"(ABeautifulGame/glTF/ABeautifulGame.gltf)", default_search_paths);
  dlss_applet->onFileDrop(scn_file);

  // Run as fast as possible, without waiting for display vertical syncs.
  app.setVsync(false);

  app.run();
  app.deinit();
  dlss_applet.reset();
  g_elem_camera.reset();
  g_dbgPrintf.reset();

  vkCtx.deinit();

  return EXIT_SUCCESS;
}
