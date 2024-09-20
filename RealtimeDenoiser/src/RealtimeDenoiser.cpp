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

//////////////////////////////////////////////////////////////////////////
/*

 This sample loads GLTF scenes and renders them using RTX (path tracer)
 
 The path tracer renders into multiple G-Buffers, which are used 
 to denoise the image using NRD.

 The denoised image is then antialiased using TAA and tone mapped to produce
 the final image.

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <filesystem>
#include <math.h>
#include <memory>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/fileoperations.hpp"
#include "nvp/nvpsystem.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/structs_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvh/gltfscene.hpp"
#include "nvvkhl/gltf_scene_rtx.hpp"
#include "nvvkhl/gltf_scene_vk.hpp"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"
#include "nvvkhl/element_dbgprintf.hpp"
#include "nvvkhl/shaders/dh_comp.h"

#include "shaders/host_device.h"
#include "_autogen/nrd.rchit.h"
#include "_autogen/nrd.rgen.h"
#include "_autogen/nrd.rmiss.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/pathtrace.rahit.h"
#include "_autogen/compositing.comp.h"
#include "_autogen/taa.comp.h"

#include "NRDWrapper.hpp"

#include <glm/gtc/type_ptr.hpp>
#include "Nrd_ui.h"

std::shared_ptr<nvvkhl::ElementCamera>    g_elem_camera;
std::shared_ptr<nvvkhl::ElementDbgPrintf> g_dbgPrintf;

using namespace nvh::gltf;

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
class NRDEngine : public nvvkhl::IAppElement
{
  enum GbufferNames
  {
    eGBufLdr,
    eGBufBaseColorMetalness = eGBufLdr,
    eGBufOutDiffRadianceHitDist,
    eGBufDiffRadianceHitDist,     // diffuse radiance and distance to first secondary hit
    eGBufSpecRadianceHitDist,     // specular radiance and distance to
    eGBufOutSpecRadianceHitDist,  //
    eGBufNormalRoughness,         // encoded worldspace normal and linear roughness
    eGBufMotionVectors,           // 2D motion vectors
    eGBufViewZ,                   // linear viewspace depth
    eGBufOutDebugView,            // NRD
    eGBufDenoisedUnpacked,
    eGBufDirectLighting,
    eGBufTaa,  // out from TAA

    eGBufNumBuffers
  };

  struct Settings
  {
    int       maxFrames{200000};
    int       maxDepth{5};
    glm::vec4 clearColor{1.F};
    float     envRotation{0.F};
    bool      showAxis{true};
  } m_settings;

public:
  NRDEngine() { m_frameInfo.clearColor = glm::vec4(1.F); };

  ~NRDEngine() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.physicalDevice         = app->getPhysicalDevice();
    allocator_info.device                 = app->getDevice();
    allocator_info.instance               = app->getInstance();
    allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);         // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(allocator_info);  // Allocator
    m_scene = std::make_unique<nvh::gltf::Scene>();                // GLTF scene
    m_sceneVk = std::make_unique<nvvkhl::SceneVk>(m_device, m_app->getPhysicalDevice(), m_alloc.get());  // GLTF Scene buffers
    m_sceneRtx = std::make_unique<nvvkhl::SceneRtx>(m_device, m_app->getPhysicalDevice(), m_alloc.get());  // GLTF Scene BLAS/TLAS
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_device, m_alloc.get());
    m_sbt        = std::make_unique<nvvk::SBTWrapper>();
    m_picker     = std::make_unique<nvvk::RayPickerKHR>(m_device, m_app->getPhysicalDevice(), m_alloc.get());
    m_vkAxis     = std::make_unique<nvvk::AxisVK>();
    m_hdrEnv     = std::make_unique<nvvkhl::HdrEnv>(m_device, m_app->getPhysicalDevice(), m_alloc.get());
    m_rtxSet     = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_sceneSet   = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_nrdSet     = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    m_hdrEnv->loadEnvironment("");

    // Requesting ray tracing properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &rt_prop;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);
    // Create utilities to create the Shading Binding Table (SBT)
    uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt->setup(m_app->getDevice(), gct_queue_index, m_alloc.get(), rt_prop);

    // Create resources
    createGbuffers(m_viewSize);
    createVulkanBuffers();

    // Axis in the bottom left corner
    nvvk::AxisVK::CreateAxisInfo ainfo;
    ainfo.colorFormat = {m_gBuffers->getColorFormat(0)};
    ainfo.depthFormat = m_gBuffers->getDepthFormat();
    m_vkAxis->init(m_device, ainfo);

    m_tonemapper->createComputePipeline();
    createCompositionPipeline();
    createTaaPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    vkDeviceWaitIdle(m_device);

    createGbuffers({width, height});

    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eGBufTaa),
                                              m_gBuffers->getDescriptorImageInfo(eGBufLdr));
    writeRtxSet();

    nvvk::Texture userTexturePool[size_t(nrd::ResourceType::MAX_NUM)] = {};

    auto poolTextureFromGBufTexture = [&](nrd::ResourceType nrdResource, GbufferNames gbufIndex) {
      userTexturePool[size_t(nrdResource)] = {m_gBuffers->getColorImage(gbufIndex), nvvk::NullMemHandle,
                                              m_gBuffers->getDescriptorImageInfo(gbufIndex)};
    };

    // #NRD Fill the user pool with our textures
    poolTextureFromGBufTexture(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, eGBufDiffRadianceHitDist);
    poolTextureFromGBufTexture(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, eGBufSpecRadianceHitDist);
    poolTextureFromGBufTexture(nrd::ResourceType::IN_NORMAL_ROUGHNESS, eGBufNormalRoughness);
    poolTextureFromGBufTexture(nrd::ResourceType::IN_MV, eGBufMotionVectors);
    poolTextureFromGBufTexture(nrd::ResourceType::IN_VIEWZ, eGBufViewZ);
    poolTextureFromGBufTexture(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, eGBufOutDiffRadianceHitDist);
    poolTextureFromGBufTexture(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, eGBufOutSpecRadianceHitDist);
    poolTextureFromGBufTexture(nrd::ResourceType::OUT_VALIDATION, eGBufOutDebugView);

    poolTextureFromGBufTexture(nrd::ResourceType::IN_SIGNAL, eGBufDiffRadianceHitDist);
    poolTextureFromGBufTexture(nrd::ResourceType::OUT_SIGNAL, eGBufOutDiffRadianceHitDist);


    m_nrd.reset(new NRDWrapper(*m_alloc, width, height, userTexturePool));
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
      auto filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
                                                      "glTF(.gltf, .glb), HDR(.hdr)|*.gltf;*.glb;*.hdr");
      onFileDrop(filename.c_str());
    }
  }

  void onFileDrop(const char* filename) override
  {
    namespace fs = std::filesystem;
    vkDeviceWaitIdle(m_device);
    std::string extension = fs::path(filename).extension().string();
    if(extension == ".gltf" || extension == ".glb")
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
    using namespace ImGuiH;

    bool reset{false};
    // Pick under mouse cursor
    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
    {
      screenPicking();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_M))
    {
      onResize(m_app->getViewportSize().width, m_app->getViewportSize().height);  // Force recreation of G-Buffers
      reset = true;
    }

    {  // Setting menu
      ImGui::Begin("Settings");

      if(ImGui::CollapsingHeader("Camera"))
      {
        ImGuiH::CameraWidget();
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
        PropertyEditor::entry("Show Axis", [&] { return ImGui::Checkbox("##4", &m_settings.showAxis); });
        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Environment"))
      {
        PropertyEditor::begin();
        if(PropertyEditor::treeNode("Hdr"))
        {
          PropertyEditor::entry(
              "Intensity",
              [&] {
                static float intensity = 1.0f;
                bool hit = ImGui::SliderFloat("##Color", &intensity, 0, 100, "%.3f", ImGuiSliderFlags_Logarithmic);
                m_settings.clearColor = glm::vec4(intensity, intensity, intensity, 1);
                return hit;
              },
              "HDR multiplier");

          PropertyEditor::entry("Rotation", [&] { return ImGui::SliderAngle("Rotation", &m_settings.envRotation); }, "Rotating the environment");
          PropertyEditor::treePop();
        }
        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        m_tonemapper->onUI();
      }

      if(ImGui::CollapsingHeader("NRD", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();

        const char* const items[] = {"ReLAX", "ReBLUR", "Reference"};
        if(PropertyEditor::entry("Method",
                                 [&]() { return ImGui::ListBox("Method", &m_pushConst.method, items, arraySize(items)); }))
        {
          reset = true;
        }

        PropertyEditor::entry("Split",
                              [&]() { return ImGui::SliderFloat("#Split", &m_nrdSettings.splitScreen, 0.0, 1.0f); });

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
        ImVec2 tumbnailSize = {100 * m_gBuffers->getAspectRatio(), 100};

        auto showBuffer = [&](const char* name, GbufferNames buffer) {
          ImGui::Text("%s", name);
          if(ImGui::ImageButton(m_gBuffers->getDescriptorSet(buffer), tumbnailSize))
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

    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eGBufTaa),
                                              m_gBuffers->getDescriptorImageInfo(eGBufLdr));


    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(m_showBuffer), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_scene->valid())
    {
      return;
    }

    updateFrame();

    auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

    // Get camera info
    float     view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    const auto& clip = CameraManip.getClipPlanes();
    m_frameInfo.view = CameraManip.getMatrix();
    m_frameInfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);

    auto unflippedProj = m_frameInfo.proj;  // There's some weirness going on with the vertical

    // Were're feeding the raytracer with a flipped matrix for convenience
    m_frameInfo.proj[1][1] *= -1;

    m_frameInfo.projInv     = glm::inverse(m_frameInfo.proj);
    m_frameInfo.viewInv     = glm::inverse(m_frameInfo.view);
    m_frameInfo.envRotation = m_settings.envRotation;
    m_frameInfo.clearColor  = m_settings.clearColor;
    m_frameInfo.jitter      = halton(m_frame) - vec2(0.5);

    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

    // Push constant
    m_pushConst.maxDepth   = m_settings.maxDepth;
    m_pushConst.frame      = m_frame;
    m_pushConst.mouseCoord = g_dbgPrintf->getMouseCoord();

    raytraceScene(cmd);

    // #NRD
    {
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
        //m_nrdSettings.motionVectorScale[0] = m_nrdSettings.motionVectorScale[1] = 1.0f;
        //m_nrdSettings.motionVectorScale[2] = 0.0f;

        m_nrdSettings.isMotionVectorInWorldSpace = true;

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
          auto poolTextureFromGBufTexture = [&](GbufferNames gbufIndex) -> nvvk::Texture {
            return {m_gBuffers->getColorImage(gbufIndex), nvvk::NullMemHandle, m_gBuffers->getDescriptorImageInfo(gbufIndex)};
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

    auto shaderWriteToShaderRead = [this](GbufferNames buffer) {
      return nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(buffer), VK_ACCESS_SHADER_WRITE_BIT,
                                          VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    };
    auto shaderReadToShaderWrite = [this](GbufferNames buffer) {
      return nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(buffer), VK_ACCESS_SHADER_READ_BIT,
                                          VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    };

    // Transition the intermediate textures for reading during final assembly.
    {
      std::vector<VkImageMemoryBarrier> barriers{
          shaderWriteToShaderRead(eGBufOutDiffRadianceHitDist), shaderWriteToShaderRead(eGBufOutSpecRadianceHitDist),
          shaderWriteToShaderRead(eGBufDirectLighting),         shaderWriteToShaderRead(eGBufNormalRoughness),
          shaderWriteToShaderRead(eGBufBaseColorMetalness),     shaderWriteToShaderRead(eGBufViewZ),
          shaderWriteToShaderRead(eGBufOutDebugView),           shaderReadToShaderWrite(eGBufDenoisedUnpacked)};

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, barriers.size(), barriers.data());
    }

    // Assemble denoised diffuse and specular radiances
    compose(cmd, m_gBuffers->getColorImageView(eGBufDenoisedUnpacked));

    {
      std::vector<VkImageMemoryBarrier> barriers{
          shaderReadToShaderWrite(eGBufOutDiffRadianceHitDist), shaderReadToShaderWrite(eGBufOutSpecRadianceHitDist),
          shaderReadToShaderWrite(eGBufDirectLighting),         shaderReadToShaderWrite(eGBufNormalRoughness),
          shaderReadToShaderWrite(eGBufBaseColorMetalness),     shaderReadToShaderWrite(eGBufViewZ),
          shaderWriteToShaderRead(eGBufDenoisedUnpacked)};

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, barriers.size(), barriers.data());
    }

    // Apply temporal aliasing
    applyTaa(cmd);

    // Apply tonemapper - take GBuffer-X and output to GBuffer-0
    m_tonemapper->runCompute(cmd, m_gBuffers->getSize());

    // Render corner axis
    renderAxis(cmd);

    m_frame++;
  }


private:
  void createScene(const std::string& filename)
  {
    if(!m_scene->load(filename))
    {
      LOGE("Error loading scene");
      return;
    }

    nvvkhl::setCamera(filename, m_scene->getRenderCameras(), m_scene->getSceneBounds());  // Camera auto-scene-fitting
    g_elem_camera->setSceneRadius(m_scene->getSceneBounds().radius());                    // Navigation help

    {  // Create the Vulkan side of the scene
      auto cmd = m_app->createTempCmdBuffer();
      m_sceneVk->create(cmd, *m_scene);
      m_sceneRtx->create(cmd, *m_scene, *m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);  // Create BLAS / TLAS

      m_app->submitAndWaitTempCmdBuffer(cmd);

      m_picker->setTlas(m_sceneRtx->tlas());
    }

    // Descriptor Set and Pipelines
    createSceneSet();
    createRtxSet();
    createNrdSet();
    createRtxPipeline();  // must recreate due to texture changes
    writeSceneSet();
    writeRtxSet();
  }

  void createGbuffers(const glm::vec2& size)
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
    m_gBuffers.reset();
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), vk_size, color_buffers, VK_FORMAT_UNDEFINED);

    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufOutDiffRadianceHitDist),
                           nrd::GetResourceTypeString(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufDiffRadianceHitDist),
                           nrd::GetResourceTypeString(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufOutSpecRadianceHitDist),
                           nrd::GetResourceTypeString(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufSpecRadianceHitDist),
                           nrd::GetResourceTypeString(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufNormalRoughness),
                           nrd::GetResourceTypeString(nrd::ResourceType::IN_NORMAL_ROUGHNESS));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufMotionVectors), nrd::GetResourceTypeString(nrd::ResourceType::IN_MV));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufViewZ), nrd::GetResourceTypeString(nrd::ResourceType::IN_VIEWZ));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufOutDebugView),
                           nrd::GetResourceTypeString(nrd::ResourceType::OUT_VALIDATION));
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufDenoisedUnpacked), "AssembledHDR");
    m_dutil->setObjectName(m_gBuffers->getColorImage(eGBufDirectLighting), "DirectLightingHDR");

    // Indicate the renderer to reset its frame
    resetFrame();
  }

  // Create all Vulkan buffer data
  void createVulkanBuffers()
  {
    auto* cmd = m_app->createTempCmdBuffer();

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createRtxSet()
  {
    auto& d = m_rtxSet;
    d->deinit();
    d->init(m_device);

    // This descriptor set, holds the top level acceleration structure and the output image
    d->addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);

    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  void createSceneSet()
  {
    auto& d = m_sceneSet;
    d->deinit();
    d->init(m_device);

    // This descriptor set, holds the top level acceleration structure and the output image
    d->addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk->nbTextures(), VK_SHADER_STAGE_ALL);
    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  void createNrdSet()
  {
    auto& d = m_nrdSet;
    d->deinit();
    d->init(m_device);

    // #NRD
    d->addBinding(NrdBindings::eNormal_Roughness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(NrdBindings::eUnfiltered_Diff, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(NrdBindings::eUnfiltered_Spec, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(NrdBindings::eViewZ, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(NrdBindings::eDirectLighting, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(NrdBindings::eObjectMotion, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(NrdBindings::eBaseColor_Metalness, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);


    d->initLayout();
    d->initPool(1);
    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet());
  }

  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    auto& p = m_rtxPipe;
    p.destroy(m_device);
    p.plines.resize(1);
    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eNrdRaygen,  // #NRD
      eMiss,
      eClosestHit,
      eAnyHit,
      eNrdHit,  // #NRD
      eNrdMiss,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point
    // Raygen
    stage.module    = nvvk::createShaderModule(m_device, nrd_rgen, sizeof(nrd_rgen));
    stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;
    // #NRD - Raygen
    stage.module       = nvvk::createShaderModule(m_device, nrd_rgen, sizeof(nrd_rgen));
    stage.stage        = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eNrdRaygen] = stage;
    // Miss
    stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
    stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;
    // #NRD - Miss
    stage.module     = nvvk::createShaderModule(m_device, nrd_rmiss, sizeof(nrd_rmiss));
    stage.stage      = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eNrdMiss] = stage;
    // Hit Group - Closest Hit
    stage.module        = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
    stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;
    // AnyHit
    stage.module    = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));
    stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eAnyHit] = stage;
    // #NRD - Closest Hit
    stage.module    = nvvk::createShaderModule(m_device, nrd_rchit, sizeof(nrd_rchit));
    stage.stage     = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eNrdHit] = stage;
    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shaderGroups.push_back(group);
    group.generalShader = eNrdRaygen;  // #NRD
    shaderGroups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shaderGroups.push_back(group);
    // #NRD Miss
    group.generalShader = eNrdMiss;
    shaderGroups.push_back(group);

    // closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.anyHitShader     = eAnyHit;
    shaderGroups.push_back(group);

    // #NRD closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eNrdHit;
    group.anyHitShader     = eAnyHit;
    shaderGroups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant)};

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges    = &push_constant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtxSet->getLayout(), m_sceneSet->getLayout(),
                                                              m_nrdSet->getLayout(), m_hdrEnv->getDescriptorSetLayout()};
    pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(rt_desc_set_layouts.size());
    pipeline_layout_create_info.pSetLayouts                = rt_desc_set_layouts.data();
    vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    ray_pipeline_info.pGroups                      = shaderGroups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 2;  // Ray depth
    ray_pipeline_info.layout                       = p.layout;
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (p.plines).data());
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt->create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
    for(auto& s : stages)
    {
      vkDestroyShaderModule(m_device, s.module, nullptr);
    }
  }

  void writeRtxSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_sceneRtx->tlas();
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    desc_as_info.accelerationStructureCount = 1;
    desc_as_info.pAccelerationStructures    = &tlas;

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eTlas, &desc_as_info));

    // #NRD images that the RTX pipeline produces
    auto bindImage = [&](NrdBindings binding, GbufferNames gbuf) {
      writes.emplace_back(m_nrdSet->makeWrite(0, binding, &m_gBuffers->getDescriptorImageInfo(gbuf)));
    };

    // #NRD
    bindImage(NrdBindings::eUnfiltered_Diff, eGBufDiffRadianceHitDist);
    bindImage(NrdBindings::eUnfiltered_Spec, eGBufSpecRadianceHitDist);
    bindImage(NrdBindings::eNormal_Roughness, eGBufNormalRoughness);
    bindImage(NrdBindings::eViewZ, eGBufViewZ);
    bindImage(NrdBindings::eObjectMotion, eGBufMotionVectors);
    bindImage(NrdBindings::eDirectLighting, eGBufDirectLighting);
    bindImage(NrdBindings::eBaseColor_Metalness, eGBufBaseColorMetalness);  // use the LDR buffer as temporary storage for the base/metalness

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }


  void writeSceneSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    auto& d = m_sceneSet;

    // Write to descriptors
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo scene_desc{m_sceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, SceneBindings::eFrameInfo, &dbi_unif));
    writes.emplace_back(d->makeWrite(0, SceneBindings::eSceneDesc, &scene_desc));
    std::vector<VkDescriptorImageInfo> diit;
    for(const auto& texture : m_sceneVk->textures())  // All texture samplers
    {
      diit.emplace_back(texture.descriptor);
    }
    writes.emplace_back(d->makeWriteArray(0, SceneBindings::eTextures, diit.data()));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  // If the camera matrix has changed, resets the frame.
  // otherwise, increments frame.
  //
  void updateFrame()
  {
    static glm::mat4 ref_cam_matrix;
    static float     ref_fov{CameraManip.getFov()};

    const auto& m   = CameraManip.getMatrix();
    const auto  fov = CameraManip.getFov();

    if(ref_cam_matrix != m || ref_fov != fov)
    {
      ref_cam_matrix = m;
      ref_fov        = fov;
    }
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
      snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms | Frame %d", PROJECT_NAME,
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
    auto* tlas = m_sceneRtx->tlas();
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
    const auto& view = CameraManip.getMatrix();
    auto        proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, 0.1F, 1000.0F);
    proj[1][1] *= -1;

    // Setting up the data to do picking
    nvvk::RayPickerKHR::PickInfo pick_info;
    pick_info.pickX          = local_mouse_pos.x;
    pick_info.pickY          = local_mouse_pos.y;
    pick_info.modelViewInv   = glm::inverse(view);
    pick_info.perspectiveInv = glm::inverse(proj);

    // Run and wait for result
    m_picker->run(cmd, pick_info);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Retrieving picking information
    nvvk::RayPickerKHR::PickResult pr = m_picker->getResult();
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
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, world_pos, up, false);

    //    auto float_as_uint = [](float f) { return *reinterpret_cast<uint32_t*>(&f); };

    // Logging picking info.
    const nvh::gltf::RenderNode& renderNode = m_scene->getRenderNodes()[pr.instanceID];
    const tinygltf::Node&        node       = m_scene->getModel().nodes[renderNode.refNodeID];

    LOGI("Node Name: %s\n", node.name.c_str());
    LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pr.primitiveID);
    LOGI(" - Render: RenderNode: %d, RenderPrim: %d\n", pr.instanceID, pr.instanceCustomIndex);
    LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
  }

  //--------------------------------------------------------------------------------------------------
  // Render the axis in the bottom left corner of the screen
  //
  void renderAxis(const VkCommandBuffer& cmd)
  {
    if(m_settings.showAxis)
    {
      float axis_size = 50.F;

      nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(eGBufLdr)},
                                       m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR);
      r_info.pStencilAttachment = nullptr;
      // Rendering the axis
      vkCmdBeginRendering(cmd, &r_info);
      m_vkAxis->setAxisSize(axis_size);
      m_vkAxis->display(cmd, CameraManip.getMatrix(), m_gBuffers->getSize());
      vkCmdEndRendering(cmd);
    }
  }

  void raytraceScene(VkCommandBuffer cmd)
  {
    auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtxSet->getSet(), m_sceneSet->getSet(), m_nrdSet->getSet(),
                                           m_hdrEnv->getDescriptorSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtxPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant), &m_pushConst);

    const auto& size = m_gBuffers->getSize();

    auto sbtRegions = m_sbt->getRegions(1);  // #NRD Using only first RayGen
    vkCmdTraceRaysKHR(cmd, &sbtRegions[0], &sbtRegions[1], &sbtRegions[2], &sbtRegions[3], size.width, size.height, 1);

    // Making sure the rendered image is ready to be used by denoiser and tonemapper
    {
      auto scope_dbg2 = m_dutil->scopeLabel(cmd, "barrier");

      auto image_memory_barrier =
          nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eGBufOutDiffRadianceHitDist), VK_ACCESS_SHADER_WRITE_BIT,
                                       VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           0, nullptr, 0, nullptr, 1, &image_memory_barrier);
    }
  }

  void createHdr(const char* filename)
  {
    m_hdrEnv = std::make_unique<nvvkhl::HdrEnv>(m_app->getDevice(), m_app->getPhysicalDevice(), m_alloc.get());

    m_hdrEnv->loadEnvironment(filename);
  }

  void destroyResources()
  {
    m_nrd.reset();

    m_alloc->destroy(m_bFrameInfo);

    m_gBuffers.reset();

    vkDestroyPipeline(m_device, m_compositionPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_compositionLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_compositionDescSetlayout, nullptr);
    vkDestroyPipeline(m_device, m_taaPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_taaLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_taaDescSetlayout, nullptr);

    m_rtxPipe.destroy(m_device);
    m_rtxSet->deinit();
    m_sceneSet->deinit();
    m_nrdSet->deinit();
    m_sbt->destroy();
    m_picker->destroy();
    m_vkAxis->deinit();
  }

  void createCompositionPipeline()
  {
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.push_back({uint32_t(CompositionBindings::eInDiff), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(CompositionBindings::eInSpec), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(CompositionBindings::eInDirect), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(CompositionBindings::eInBaseColor_Metalness), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                              1, VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(CompositionBindings::eInNormal_Roughness), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                              VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(CompositionBindings::eInViewZ), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(CompositionBindings::eInFrameInfo), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                              VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(CompositionBindings::eCompImage), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT});

    VkDescriptorSetLayoutCreateInfo layoutInfo(nvvk::make<VkDescriptorSetLayoutCreateInfo>());

    layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    layoutInfo.bindingCount = layoutBindings.size();
    layoutInfo.pBindings    = layoutBindings.data();

    NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_compositionDescSetlayout));
    m_dutil->setObjectName(m_compositionDescSetlayout, "Composition Descriptor Set Layout");

    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(nvvk::make<VkPipelineLayoutCreateInfo>());
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts    = &m_compositionDescSetlayout;

    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &push_constant;

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_compositionLayout));

    VkShaderModuleCreateInfo shaderInfo(nvvk::make<VkShaderModuleCreateInfo>());
    shaderInfo.codeSize = sizeof(compositing_comp);
    shaderInfo.pCode    = compositing_comp;

    VkShaderModule assembleShader = VK_NULL_HANDLE;
    NVVK_CHECK(vkCreateShaderModule(m_device, &shaderInfo, nullptr, &assembleShader));

    VkPipelineShaderStageCreateInfo stageCreateInfo(nvvk::make<VkPipelineShaderStageCreateInfo>());
    stageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCreateInfo.module = assembleShader;
    stageCreateInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo(nvvk::make<VkComputePipelineCreateInfo>());
    pipelineInfo.layout = m_compositionLayout;
    pipelineInfo.stage  = stageCreateInfo;


    NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_compositionPipeline));

    m_dutil->setObjectName(m_compositionPipeline, "Composition Pipeline");

    vkDestroyShaderModule(m_device, assembleShader, nullptr);
  }

  void createTaaPipeline()
  {
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.push_back({uint32_t(TaaBindings::eInImage), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT});
    layoutBindings.push_back({uint32_t(TaaBindings::eOutImage), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT});
    VkDescriptorSetLayoutCreateInfo layoutInfo(nvvk::make<VkDescriptorSetLayoutCreateInfo>());

    layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    layoutInfo.bindingCount = layoutBindings.size();
    layoutInfo.pBindings    = layoutBindings.data();

    NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_taaDescSetlayout));
    m_dutil->setObjectName(m_taaDescSetlayout, "TAA Descriptor Set Layout");

    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(float)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(nvvk::make<VkPipelineLayoutCreateInfo>());
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts    = &m_taaDescSetlayout;

    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &push_constant;

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_taaLayout));

    VkShaderModuleCreateInfo shaderInfo(nvvk::make<VkShaderModuleCreateInfo>());
    shaderInfo.codeSize = sizeof(taa_comp);
    shaderInfo.pCode    = taa_comp;

    VkShaderModule assembleShader = VK_NULL_HANDLE;
    NVVK_CHECK(vkCreateShaderModule(m_device, &shaderInfo, nullptr, &assembleShader));

    VkPipelineShaderStageCreateInfo stageCreateInfo(nvvk::make<VkPipelineShaderStageCreateInfo>());
    stageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCreateInfo.module = assembleShader;
    stageCreateInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo(nvvk::make<VkComputePipelineCreateInfo>());
    pipelineInfo.layout = m_taaLayout;
    pipelineInfo.stage  = stageCreateInfo;


    NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_taaPipeline));

    m_dutil->setObjectName(m_taaPipeline, "TAA Pipeline");

    vkDestroyShaderModule(m_device, assembleShader, nullptr);
  }


  void compose(VkCommandBuffer& commandBuffer, VkImageView outImage)
  {
    std::vector<VkWriteDescriptorSet> writes;

    VkDescriptorImageInfo outImageInfo = {VK_NULL_HANDLE, outImage, VK_IMAGE_LAYOUT_GENERAL};
    {
      VkWriteDescriptorSet descriptorWrite(nvvk::make<VkWriteDescriptorSet>());
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      descriptorWrite.dstBinding      = uint32_t(CompositionBindings::eCompImage);
      descriptorWrite.pImageInfo      = &outImageInfo;

      writes.push_back(descriptorWrite);
    }

    VkDescriptorBufferInfo bufferInfo = {m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    {
      VkWriteDescriptorSet descriptorWrite(nvvk::make<VkWriteDescriptorSet>());
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrite.dstBinding      = uint32_t(CompositionBindings::eInFrameInfo);
      descriptorWrite.pBufferInfo     = &bufferInfo;

      writes.push_back(descriptorWrite);
    }

    auto bindImage = [&](CompositionBindings binding, GbufferNames gbufImage) {
      VkWriteDescriptorSet descriptorWrite(nvvk::make<VkWriteDescriptorSet>());
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      descriptorWrite.dstBinding      = uint32_t(binding);
      descriptorWrite.pImageInfo      = &m_gBuffers->getDescriptorImageInfo(uint32_t(gbufImage));

      writes.emplace_back(descriptorWrite);
    };

    bindImage(CompositionBindings::eInDiff, eGBufOutDiffRadianceHitDist);
    bindImage(CompositionBindings::eInSpec, eGBufOutSpecRadianceHitDist);
    bindImage(CompositionBindings::eInDirect, eGBufDirectLighting);
    bindImage(CompositionBindings::eInBaseColor_Metalness, eGBufLdr);
    bindImage(CompositionBindings::eInNormal_Roughness, eGBufNormalRoughness);
    bindImage(CompositionBindings::eInViewZ, eGBufViewZ);

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositionLayout, 0, writes.size(),
                              writes.data());

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositionPipeline);

    VkExtent2D group_counts = getGroupCounts(m_gBuffers->getSize());
    vkCmdDispatch(commandBuffer, group_counts.width, group_counts.height, 1);
  }


  void applyTaa(VkCommandBuffer& commandBuffer)
  {
    std::vector<VkWriteDescriptorSet> writes;

    auto bindImage = [&](TaaBindings binding, GbufferNames gbufImage) {
      VkWriteDescriptorSet descriptorWrite(nvvk::make<VkWriteDescriptorSet>());
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      descriptorWrite.dstBinding      = uint32_t(binding);
      descriptorWrite.pImageInfo      = &m_gBuffers->getDescriptorImageInfo(uint32_t(gbufImage));

      writes.emplace_back(descriptorWrite);
    };

    bindImage(TaaBindings::eInImage, eGBufDenoisedUnpacked);
    bindImage(TaaBindings::eOutImage, eGBufTaa);

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_taaLayout, 0, writes.size(), writes.data());

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_taaPipeline);
    float alpha = 0.1F;
    vkCmdPushConstants(commandBuffer, m_taaLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &alpha);

    VkExtent2D group_counts = getGroupCounts(m_gBuffers->getSize());
    vkCmdDispatch(commandBuffer, group_counts.width, group_counts.height, 1);
  }


  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::unique_ptr<nvvkhl::AllocVma> m_alloc;

  glm::vec2                                     m_viewSize = {1, 1};
  VkDevice                                      m_device   = VK_NULL_HANDLE;
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;  // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtxSet;    // Descriptor set
  std::unique_ptr<nvvk::DescriptorSetContainer> m_sceneSet;  // Descriptor set
  std::unique_ptr<nvvk::DescriptorSetContainer> m_nrdSet;    // Descriptor set

  // Resources
  nvvk::Buffer m_bFrameInfo;

  // Pipeline
  RtxPushConstant m_pushConst{
      -1,          // frame
      10.f,        // magic-scene number
      7,           // max ray recursion
      NRD_REBLUR,  // method,
      1.0,         // meterToUnitsMultiplier
      -1.0,        // overrideRoughness
      -1.0,        // overrideMetallic
  };  // Information sent to the shader
  nvvkhl::PipelineContainer m_rtxPipe;
  int                       m_frame{0};
  FrameInfo                 m_frameInfo{};

  GbufferNames m_showBuffer = eGBufLdr;

  std::unique_ptr<nvh::gltf::Scene>              m_scene;
  std::unique_ptr<nvvkhl::SceneVk>               m_sceneVk;
  std::unique_ptr<nvvkhl::SceneRtx>              m_sceneRtx;
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;
  std::unique_ptr<nvvk::SBTWrapper>              m_sbt;     // Shading binding table wrapper
  std::unique_ptr<nvvk::RayPickerKHR>            m_picker;  // For ray picking info
  std::unique_ptr<nvvk::AxisVK>                  m_vkAxis;
  std::unique_ptr<nvvkhl::HdrEnv>                m_hdrEnv;

  // #NRD
  std::unique_ptr<NRDWrapper> m_nrd;
  nrd::CommonSettings         m_nrdSettings    = {};
  nrd::RelaxSettings          m_relaxSettings  = {};
  nrd::ReblurSettings         m_reblurSettings = {};

  // Assemble compute shader
  VkPipeline            m_compositionPipeline      = {};
  VkPipelineLayout      m_compositionLayout        = {};
  VkDescriptorSetLayout m_compositionDescSetlayout = VK_NULL_HANDLE;
  VkPipeline            m_taaPipeline              = {};
  VkPipelineLayout      m_taaLayout                = {};
  VkDescriptorSetLayout m_taaDescSetlayout         = VK_NULL_HANDLE;
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int /*argc*/, char** /*argv*/) -> int
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name  = PROJECT_NAME " Example";
  spec.vSync = true;

  nvvk::ContextCreateInfo ctxInfo;
  ctxInfo.apiMajor = 1;
  ctxInfo.apiMinor = 3;

  ctxInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
  ctxInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &ray_query_features);  // Used for picking
  ctxInfo.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  ctxInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeature);
  ctxInfo.addDeviceExtension(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, false);

  // Display extension
  ctxInfo.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  ctxInfo.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  nvvkhl::addSurfaceExtensions(ctxInfo.instanceExtensions);

  g_dbgPrintf                   = std::make_shared<nvvkhl::ElementDbgPrintf>();
  ctxInfo.instanceCreateInfoExt = g_dbgPrintf->getFeatures();

  nvvk::Context vkCtx;
  if(!vkCtx.init(ctxInfo))
  {
    LOGE("ERROR: Vulkan Context Creation failed.");
    return EXIT_FAILURE;
  }

  spec.instance       = vkCtx.m_instance;
  spec.physicalDevice = vkCtx.m_physicalDevice;
  spec.device         = vkCtx.m_device;
  spec.queues.push_back({vkCtx.m_queueGCT.familyIndex, vkCtx.m_queueGCT.queueIndex, vkCtx.m_queueGCT.queue});

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create application elements
  auto nrd_denoiser = std::make_shared<NRDEngine>();
  g_elem_camera     = std::make_shared<nvvkhl::ElementCamera>();

  app->addElement(g_elem_camera);
  app->addElement(nrd_denoiser);
  app->addElement(g_dbgPrintf);
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit

  // Search paths
  std::vector<std::string> default_search_paths = {".", "..", "../..", "../../.."};

  // Load scene
  std::string scn_file = nvh::findFile(R"(media/cornellBox.gltf)", default_search_paths, true);
  nrd_denoiser->onFileDrop(scn_file.c_str());

  // Load HDR
  std::string hdr_file = nvh::findFile(R"(media/spruit_sunrise_1k.hdr)", default_search_paths, true);
  nrd_denoiser->onFileDrop(hdr_file.c_str());

  // Run as fast as possible, without waiting for display vertical syncs.
  app->setVsync(false);

  app->run();
  nrd_denoiser.reset();
  app.reset();

  return EXIT_SUCCESS;
}
