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
#pragma once
#include "imgui/imgui_helper.h"
#include <NRD.h>

namespace Nrd_ui {
static void render(nrd::ReblurSettings& reblurSettings, nrd::RelaxSettings& relaxSettings)
{
  namespace PE = ImGuiH::PropertyEditor ;
  if(PE::treeNode("ReBlUR"))
  {
    PE::entry(
        "Max Accumulated Frame Num",
        [&]() { return ImGui::SliderInt("#Reblur Radius", (int*)&reblurSettings.maxAccumulatedFrameNum, 0, 60); },
        "maximum number of linearly accumulated frames (= FPS * time of accumulation )");

    PE::entry(
        "Max Fast Accumulated Frame Num",
        [&]() { return ImGui::SliderInt("#Relax Radius", (int*)&reblurSettings.maxFastAccumulatedFrameNum, 0, 60); },
        "maximum number of linearly accumulated frames in fast history (less than maxAccumulatedFrameNum )");

    PE::entry(
        "History Fix Frame Num",
        [&]() { return ImGui::SliderInt("#Relax Radius", (int*)&reblurSettings.historyFixFrameNum, 0, 3); },
        "number of reconstructed frames after history reset (less than maxFastAccumulatedFrameNum)");

    PE::entry(
        "Diffuse Prepass Blur Radius",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.diffusePrepassBlurRadius, 0.0f, 100.0f);
        },
        "pre-accumulation spatial reuse pass blur radius");


    PE::entry(
        "Specular Prepass Blur Radius",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.specularPrepassBlurRadius, 0.0f, 100.0f);
        },
        "pre-accumulation spatial reuse pass blur radius");

    PE::entry(
        "Min Base Blur Radius",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.minBlurRadius, 0.0f, 100.0f); },
        "(pixels) - min denoising radius (for converged state)");

    PE::entry(
        "Max Base Blur Radius",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.maxBlurRadius, 0.0f, 100.0f); },
        "(pixels) - max denoising radius (gets reduced over time, 30 is a baseline for 1440p)");

    PE::entry(
        "Lobe Angle Fraction",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.lobeAngleFraction, 0.0f, 1.0f); },
        "base fraction of diffuse or specular lobe angle used to drive normal based rejection");

    PE::entry(
        "Roughness Fraction",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.roughnessFraction, 0.0f, 1.0f); },
        "base fraction of center roughness used to drive roughness based rejection");

    PE::entry(
        "Roughness Threshold",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.responsiveAccumulationRoughnessThreshold, 0.0f, 1.0f);
        },
        "if roughness < this, temporal accumulation becomes responsive and driven by roughness (useful for animated water)");

    PE::entry(
        "Stabilization Strength",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.stabilizationStrength, 0.0f, 1.0f); },
        "stabilizes output, more stabilization improves antilag (clean signals can use lower values)");

    PE::entry(
        "Plane Distance Sensitivity",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&reblurSettings.planeDistanceSensitivity, 0.0f, 1.0f);
        },
        "represents maximum allowed deviation from local tangent plane");

    PE::entry(
        "Enable Anti-Firefly",
        [&]() { return ImGui::Checkbox("##Enable Anti-Firefly", &reblurSettings.enableAntiFirefly); },
        "Adds bias in case of badly defined signals, but tries to fight with fireflies");

    PE::entry(
        "Enable Performance Mode",
        [&]() { return ImGui::Checkbox("##Enable Performance Mode", &reblurSettings.enablePerformanceMode); },
        "Boosts performance by sacrificing IQ");

    PE::entry(
        "Enable Material Test For Diffuse",
        [&]() {
          return ImGui::Checkbox("##Enable Material Test For Diffuse", &reblurSettings.enableMaterialTestForDiffuse);
        },
        "Spatial passes do optional material index comparison as: ( materialEnabled ? material[ center ] == material[ sample ] : 1 )");

    PE::entry(
        "Enable Material Test For Specular",
        [&]() {
          return ImGui::Checkbox("##Enable Material Test For Specular", &reblurSettings.enableMaterialTestForSpecular);
        },
        "Spatial passes do optional material index comparison as: ( materialEnabled ? material[ center ] == material[ sample ] : 1 )");


    PE::entry(
        "Use Prepass Only For Specular Motion Estimation",
        [&]() {
          return ImGui::Checkbox("##Use Prepass Only For Specular Motion Estimation",
                                 &reblurSettings.usePrepassOnlyForSpecularMotionEstimation);
        },
        R"(In rare cases, when bright samples are so sparse that any other bright neighbor can't
be reached, pre-pass transforms a standalone bright pixel into a standalone bright blob,
worsening the situation. Despite that it's a problem of sampling, the denoiser needs to
handle it somehow on its side too. Diffuse pre-pass can be just disabled, but for specular
it's still needed to find optimal hit distance for tracking. This boolean allow to use
specular pre-pass for tracking purposes only)");

    PE::treePop();
  }


  if(PE::treeNode("ReLAX"))
  {
    if(PE::treeNode("Anti-Lag"))
    {
      PE::entry(
          "Acceleration Amount",
          [&]() {
            return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.antilagSettings.accelerationAmount, 0.0f, 1.0f);
          },
          "amount of history acceleration if history clamping happened in pixel");

      PE::entry(
          "Spatial Sigma Scale",
          [&]() {
            return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.antilagSettings.spatialSigmaScale, 0.0f, 10.0f);
          },
          "amount of history reset, 0.0 - no reset, 1.0 - full reset");

      PE::entry(
          "Temporal Sigma Scale",
          [&]() {
            return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.antilagSettings.temporalSigmaScale, 0.0f, 10.0f);
          },
          "amount of history reset, 0.0 - no reset, 1.0 - full reset");

      PE::entry(
          "Reset Amount",
          [&]() {
            return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.antilagSettings.resetAmount, 0.0f, 1.0f);
          },
          "amount of history reset, 0.0 - no reset, 1.0 - full reset");

      PE::treePop();
    }

    PE::entry(
        "Diffuse Prepass Blur Radius",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.diffusePrepassBlurRadius, 0.0f, 100.0f);
        },
        "pre-accumulation spatial reuse pass blur radius (0 = disabled, must be used in case of probabilistic sampling)");

    PE::entry(
        "Specular Prepass Blur Radius",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.specularPrepassBlurRadius, 0.0f, 100.0f);
        },
        "pre-accumulation spatial reuse pass blur radius (0 = disabled, must be used in case of probabilistic sampling)");

    PE::entry(
        "Diffuse Max Accumulated Frame Num",
        [&]() { return ImGui::SliderInt("#Relax Radius", (int*)&relaxSettings.diffuseMaxAccumulatedFrameNum, 0, 60); },
        "maximum number of linearly accumulated frames (= FPS * time of accumulation )");

    PE::entry(
        "Specular Max Accumulated Frame Num",
        [&]() { return ImGui::SliderInt("#Relax Radius", (int*)&relaxSettings.specularMaxAccumulatedFrameNum, 0, 60); },
        "maximum number of linearly accumulated frames (= FPS * time of accumulation )");

    PE::entry(
        "Diffuse Max Fast Accumulated Frame Num",
        [&]() {
          return ImGui::SliderInt("#Relax Radius", (int*)&relaxSettings.diffuseMaxFastAccumulatedFrameNum, 0, 60);
        },
        "maximum number of linearly accumulated frames in fast history (less than maxAccumulatedFrameNum )");

    PE::entry(
        "Specular Max Fast Accumulated Frame Num",
        [&]() {
          return ImGui::SliderInt("#Relax Radius", (int*)&relaxSettings.specularMaxFastAccumulatedFrameNum, 0, 60);
        },
        "maximum number of linearly accumulated frames in fast history (less than maxAccumulatedFrameNum )");


    uint32_t historyFixFrameNum = 3;
    PE::entry(
        "History Fix Frame Num",
        [&]() { return ImGui::SliderInt("#Relax Radius", (int*)&relaxSettings.historyFixFrameNum, 0, 3); },
        "number of reconstructed frames after history reset (less than maxFastAccumulatedFrameNum)");

    PE::entry(
        "Diffuse Phi Luminance",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.diffusePhiLuminance, 0.0f, 3.0f); },
        "A-trous edge stopping Luminance sensitivity");

    PE::entry(
        "Specular Phi Luminance",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.specularPhiLuminance, 0.0f, 3.0f); },
        "A-trous edge stopping Luminance sensitivity");


    PE::entry(
        "Diffuse Lobe Angle Fraction",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.diffuseLobeAngleFraction, 0.0f, 1.0f);
        },
        "base fraction of diffuse or specular lobe angle used to drive normal based rejection");

    PE::entry(
        "Specular Lobe Angle Fraction",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.specularLobeAngleFraction, 0.0f, 1.0f);
        },
        "base fraction of diffuse or specular lobe angle used to drive normal based rejection");

    PE::entry(
        "Roughness Fraction",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.roughnessFraction, 0.0f, 1.0f); },
        "base fraction of center roughness used to drive roughness based rejection");

    PE::entry(
        "Specular Variance Boost",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.specularVarianceBoost, 0.0f, 10.0f); },
        "how much variance we inject to specular if reprojection confidence is low");

    PE::entry(
        "Specular Lobe Angle Slack",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.specularLobeAngleSlack, 0.0f, 1.0f); },
        "slack for the specular lobe angle used in normal based rejection of specular during A-Trous passes");

    PE::entry(
        "History Fix Edge Stopping Normal Power",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.historyFixEdgeStoppingNormalPower, 0.0f, 10.0f);
        },
        "normal edge stopper for history reconstruction pass");

    PE::entry(
        "History Clamping Color Box Sigma Scale",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.historyClampingColorBoxSigmaScale, 1.0f, 3.0f);
        },
        "standard deviation scale of color box for clamping main slow history to responsive fast history");

    PE::entry(
        "Spatial Variance Estimation History Threshold",
        [&]() {
          return ImGui::SliderInt("#Relax Radius", (int*)&relaxSettings.spatialVarianceEstimationHistoryThreshold, 0, 10);
        },
        "history length threshold below which spatial variance estimation will be executed");

    PE::entry(
        "Atrous Iteration Num",
        [&]() { return ImGui::SliderInt("#Relax Radius", (int*)&relaxSettings.atrousIterationNum, 2, 8); },
        "number of iterations for A-Trous wavelet transform");

    PE::entry(
        "Diffuse Min Luminance Weight",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.diffuseMinLuminanceWeight, 0.0f, 1.0f);
        },
        "A-trous edge stopping Luminance weight minimum");

    PE::entry(
        "Specular Min Luminance Weight",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.specularMinLuminanceWeight, 0.0f, 1.0f);
        },
        "A-trous edge stopping Luminance weight minimum");

    PE::entry(
        "Depth Threshold",
        [&]() { return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.depthThreshold, 0.0f, 0.01f); },
        "Depth threshold for spatial passes");

    PE::entry(
        "Confidence Driven Relaxation Multiplier",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.confidenceDrivenRelaxationMultiplier, 0.0f, 1.0f);
        },
        "Confidence inputs can affect spatial blurs, relaxing some weights in areas with low confidence");

    PE::entry(
        "Confidence Driven Luminance Edge Stopping Relaxation",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius",
                                    (float*)&relaxSettings.confidenceDrivenLuminanceEdgeStoppingRelaxation, 0.0f, 1.0f);
        },
        "Confidence inputs can affect spatial blurs, relaxing some weights in areas with low confidence");

    PE::entry(
        "Confidence Driven Normal Edge Stopping Relaxation",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.confidenceDrivenNormalEdgeStoppingRelaxation, 0.0f, 1.0f);
        },
        "Confidence inputs can affect spatial blurs, relaxing some weights in areas with low confidence");

    PE::entry(
        "Luminance Edge Stopping Relaxation",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f);
        },
        "How much we relax roughness based rejection for spatial filter in areas where specular reprojection is low");

    PE::entry(
        "Normal Edge Stopping Relaxation",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.normalEdgeStoppingRelaxation, 0.0f, 1.0f);
        },
        "How much we relax rejection for spatial filter based on roughness and view vector");

    PE::entry(
        "Roughness Edge Stopping Relaxation",
        [&]() {
          return ImGui::SliderFloat("#Relax Radius", (float*)&relaxSettings.roughnessEdgeStoppingRelaxation, 0.0f, 1.0f);
        },
        "How much we relax rejection for spatial filter based on roughness and view vector");


    // Firefly suppression
    PE::entry("Enable Anti-Firefly",
              [&]() { return ImGui::Checkbox("##Enable Anti-Firefly", &relaxSettings.enableAntiFirefly); });
    PE::entry("Enable Roughness Edge Stopping", [&]() {
      return ImGui::Checkbox("##Enable Roughness Edge Stopping", &relaxSettings.enableRoughnessEdgeStopping);
    });
    PE::entry("Enable Material Test For Diffuse", [&]() {
      return ImGui::Checkbox("##Enable Material Test For Diffuse", &relaxSettings.enableMaterialTestForDiffuse);
    });
    PE::entry("Enable Material Test For Specular", [&]() {
      return ImGui::Checkbox("##Enable Material Test For Specular", &relaxSettings.enableMaterialTestForSpecular);
    });

    PE::treePop();
  }
}
}  // namespace Nrd_ui
