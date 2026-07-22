# Integration of NRD to a Vulkan application

A screenshot of the sample. It shows a path-traced glTF scene, denoised using NRD.

This example demonstrates [NRD, "NVIDIA Real-Time Denoisers",](https://github.com/NVIDIA-RTX/NRD)
 in a simple path
tracer rendering a glTF scene. NRD is a spatio-temporal post-processing library
that removes noise from Monte-Carlo based path tracers. NRD is not just a single
denoiser, in fact it is a collection of specialized denoisers for specific
kinds of data, like diffuse and specular images, ambient occlusion, and shadow
data.

NRD itself provides a comprehensive sample at [https://github.com/NVIDIA-RTX/NRD-Sample](https://github.com/NVIDIA-RTX/NRD-Sample),
showing all the bells and whistles of its various denoisers. This sample
instead focuses on an integration of the NRD library with a Vulkan renderer,
showing a simple pathtracer using NRD to denoise the rendered image.

More Resources:

- Get more information at: [https://developer.nvidia.com/nvidia-rt-denoiser](https://developer.nvidia.com/nvidia-rt-denoiser)
- NRD Sample: [https://github.com/NVIDIA-RTX/NRD-Sample](https://github.com/NVIDIA-RTX/NRD-Sample)
- NRD SDK: [https://github.com/NVIDIA-RTX/NRD](https://github.com/NVIDIA-RTX/NRD)

## Building

This sample is built on [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2),
NVIDIA's framework of Vulkan helper classes. At configuration time CMake
(`cmake/FindNvproCore2.cmake`) looks for an existing nvpro_core2 checkout next
to the sample (or under the build's `_deps` directory); if none is found it
clones it automatically (controlled by the `NVPROCORE2_DOWNLOAD` option, which
is ON by default). Otherwise, just follow the general CMake process to
configure and build the sample.

This project needs the NRD SDK. CMake for the sample project is configured to
automatically download a pinned version of NRD from
[https://github.com/NVIDIA-RTX/NRD](https://github.com/NVIDIA-RTX/NRD).

## NRD Denoising Methods

NRD offers these denoising methods:

- [ReBLUR](https://developer.nvidia.com/nvidia-rt-denoiser#REBLUR): Blur the information over multiple frames; good for low samples per pixel.
- [ReLAX](https://developer.nvidia.com/nvidia-rt-denoiser#RELAX): Preserves lighting details better.
- [SIGMA](https://developer.nvidia.com/nvidia-rt-denoiser#SIGMA): Works on all kinds of shadows.

In our sample, we test ReBLUR and ReLAX.

## CMake Integration of NRD

The NRD SDK comes with direct CMake support. In this sample we use the
'FetchContent' CMake module to automatically download and express a dependency
on the NRD SDK:

```cmake
# CMake module FetchContent
include(FetchContent)

# Tell CMake where to find NRDSDK, how to get it and where to place it locally.
# CMake will automatically look for CMakeLists.txt in there
FetchContent_Declare(
    NRDSDK
    GIT_REPOSITORY https://github.com/NVIDIA-RTX/NRD.git
    GIT_TAG v4.17.2
    GIT_SHALLOW
    GIT_SUBMODULES
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals/nrd
)

# Configure the NRD build: produce only SPIR-V binaries (no DXIL/DXBC) and link
# NRD as a static library into the executable.
option (NRD_EMBEDS_DXIL_SHADERS "NRD embeds DXIL shaders" OFF)
option (NRD_EMBEDS_DXBC_SHADERS "NRD embeds DXBC shaders" OFF)
option (NRD_STATIC_LIBRARY "NRD linked as static library into executable" ON)

# Cause CMake to download and/or update our local copy of NRDSDK.
FetchContent_MakeAvailable(NRDSDK)

# Make our project link with NRD
target_link_libraries(${PROJECT_NAME} PRIVATE NRD)
```

The NRD compile time options we set in this sample are
`NRD_EMBEDS_DXIL_SHADERS` and `NRD_EMBEDS_DXBC_SHADERS` (both disabled, so only
SPIR-V is produced) and `NRD_STATIC_LIBRARY` (enabled). Refer to
[externals/nrd/CMakeLists.txt](externals/nrd/CMakeLists.txt) for a list of
supported options, such as normal and roughness encoding. Our sample leaves
those at default (i.e. linear roughness and RGB10A2 normal encoding).

'externals/nrd/Shaders/Include/NRD.hlsli' will be used to communicate the
chosen normal and roughness encoding to the user's shaders.

## Integration of NRD

NRD itself is platform-agnostic and thus can be used with DirectX and Vulkan.
Its internal compute shaders are written in HLSL and typically compiled as part
of the NRD SDK compilation process.

There are several ways to integrate NRD.

- As a 'white-box library': the application is asked to do everything, including
allocating needed resources AND compiling NRD's shaders
- As a 'black-box library': the application creates all resources, but the
shaders will be handed to the application as precompiled binary blobs
(SPIR-V, DXIL, etc)
- Via NRD's integration layer: NRD SDK provides an integration wrapper based
on [NRI (NVIDIA Render Interface)](https://github.com/NVIDIA-RTX/NRI)

In this sample we chose the 'black-box library' approach, using a thin wrapper
around NRD that handles the integration of NRD with Vulkan.

### NRD and Slang

Even though the NRD-internal shaders are all precompiled, a few shader
fragments will be needed in the user's shaders to encode data in a way NRD
requires. Slang is directly compatible with HLSL, so no changes have to be made
when including NRD.hlsli.

NRD build is using a SPIR-V-enabled DirectX shader compiler "dxc" to
compile its HLSL shaders into SPIR-V that Vulkan can digest. dxc comes with the
Vulkan SDK, which can be downloaded [here](https://vulkan.lunarg.com/).
Another option is to download
[https://github.com/microsoft/DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler)
and build it from source.

If NRD can't find dxc at CMake configuration time, consider providing CMake
with `-DNRD_DXC_CUSTOM_PATH=<custom/path/to/dxc>` to point it to the dxc
executable directly.

### NRD and GLSL

This sample uses Slang for its own shaders (see above). For completeness: NRD
is naturally written in HLSL but can also be instructed to emit GLSL-compatible
syntax by defining `NRD_GLSL 1` before including `NRD.hlsli`. This sample does
not use that path.

### NRDWrapper

The NRD SDK comes with example code to integrate NRD with NRI (NVIDIA Render
Interface). However, vk_denoise_nrd is targeted at integration of NRD with
Vulkan specifically, utilizing
[nvpro_core2's](https://github.com/nvpro-samples/nvpro_core2) helper classes.

NRD calls this integration variant *black-box library integration.* A small
wrapper class, `NRDWrapper`, contains all the necessary code for that. Since
NRD is graphics API agnostic (supporting D3D and Vulkan), it describes
required resources in an API independent manner. NRDWrapper hides the
complexities of interpreting these descriptions and creating all resources
(buffers, textures) that NRD needs. The wrapper also interprets NRD's
description of how to perform the denoising pipeline and translates this into
executing a series of compute shader invocations with proper pipeline barriers
in-between the passes. Code comments in NRDWrapper.cpp/.hpp describe the details
of its API and implementation. Additionally, one can find a lot of information
in the NRD headers themselves. Notably, `NRDDesc.h` and `NRD.hlsli` are worth
looking into to understand the various NRD inputs and outputs as well as
learning about what resource types are needed for each type of denoiser.

NRDWrapper creates all NRD-internal textures on its own, with the exception of
the "user texture pool". This is a pool of textures that both NRD and the application
operate on. NRD calls these "resources". The user texture pool array may be sparsely
populated, depending on the used denoiser. Consult `NRDDescs.h` to see which denoisers
require which types of resource (i.e. texture) and what format each resource needs
to have.

This sample focuses on using the `REBLUR_DIFFUSE_SPECULAR` and
`RELAX_DIFFUSE_SPECULAR` denoisers, which each take separate noisy
diffuse+hitdistance and specular+hitdistance buffers as input.

## The path tracing pipeline

1. The primary driver of the rendering is the `nrd.slang` ray generation shader.
  It'll cast the primary as well as all secondary ray segments.
   This approach is preferred as it keeps the required "stack memory"
   lower than recursively casting rays from closest-hit shaders.
2. The first step for each ray is to find the first non-mirrored surface, the *primary hit*.
  At the primary hit we record all material properties (such as normals) that NRD needs and write
   them into the appropriate buffers. If we did not hit anything, the ray missed all
   geometry and thus will be recorded as hitting the skybox only. Mirrored surfaces
   need special treatment ([PSR](#mirror-like-surfaces), described below).
3. Find direct lighting contributions. For this, we loop through
  all lights, tracing a shadow ray for each. If the light is visible, we add its
   contribution to the direct lighting. We also add the primary hit's material
   light emission here. The direct light contributions are not noisy and thus
   are written into a separate buffer which does not undergo denoising.
   It will later be added back to the scene during compositing.
4. Probe the environment map lighting at the primary hit. The environment map
  has been preprocessed to provide a PDF and a randomized sampling pattern
   according to that PDF, emphasizing sampling density around its bright spots. A
   shadow ray tests if the primary hitposition is actually visible to the
   environment map in the chosen direction. Since this sampling is randomized, we
   later add its diffuse and specular environment map contribution to the noisy
   diffuse and specular output images for the denoiser to deal with. Environment
   map contributions via direct and indirect probing need to be carefully
   weighted; see the [MIS weighting](#mis-weighting) section.
5. From the hit position we perform path tracing to collect the incoming indirect light -
  once along the material's diffuse BSDF, and once along its specular BSDF.
   For this tracing we use the pathtrace_rchit.slang shader. At each hit position we do the following:
  - If the ray missed all geometry, we hit the environment map. Return its light contribution,
  multiply it by a [MIS weight](#mis-weighting), and report the end of this ray via a negative hit distance (`payload.hitT = -NRD_INF`).
  - When hitting geometry, collect direct lighting contributions just as we did for the primary ray
  (lights, envmap - see above).
  - Sample the material's BSDF for a suggested direction to follow for the next path segment.
  - In case of an absorption event, end the path by negating the hit distance (`payload.hitT = -payload.hitT`). The negative sign stops the trace while preserving the distance magnitude that NRD needs.
  - Otherwise, return a new ray segment (origin, direction and PDF value for the direction) and
  the light contribution at the current hit position to `nrd.slang`, which continues to accumulate
  the light along the path.
  - Repeat tracing paths until the ray ends or reaches the maximum number of bounces.
  Pass the distance from the primary hit to the subsequent hit to NRD.
6. De-modulate diffuse and specular color and store them encoded for the denoiser. Also store enough information
  to recompute the [(de)modulation values](#demodulation) in the composition shader.

A diagram of an application rendering pipeline using NRD. A ray tracer outputs
diffuse, specular, normal, ViewZ, and motion vectors, along with direct lighting.
The first 5 are passed to NRD, which outputs diffuse and specular. A Compose
step composites these together, and then a final TAA step performs antialiasing.

## NRD Details

As mentioned above, in this sample, we're making use of two denoisers:
`REBLUR_DIFFUSE_SPECULAR` and `RELAX_DIFFUSE_SPECULAR`.

According to NRDDescs.h, `REBLUR_DIFFUSE_SPECULAR` needs the following inputs:

- `IN_DIFF_RADIANCE_HITDIST`: a texture containing the 'demodulated' diffuse radiance at the primary hit point as well as the distance to the second hit (following the diffuse BSDF)
- `IN_SPEC_RADIANCE_HITDIST`: a texture containing the 'demodulated' specular radiance at the primary hit point as well as the distance to the second hit (following the specular BSDF)
- `MOTION`: a texture of 3D motion vectors describing each pixel's movement from the previous to the current frame. This sample renders a static scene, so it writes zero **object** motion and sets `isMotionVectorInWorldSpace = true` (with `motionVectorScale = {1, 1, 0}`). NRD still reconstructs **camera** motion on its own from the previous/current world-to-view and view-to-clip matrices we provide each frame. Note that zero motion vectors are only correct because nothing in the scene moves; animated geometry would require real motion vectors to avoid ghosting.
- `NORMAL_ROUGHNESS`: Normal, material roughness, and material ID packed into a 4D vector. The normal vector should be between [-1..1] and roughness in linear space. See the function `NRD_FrontEnd_PackNormalAndRoughness()` to see how this is done.
- `VIEWZ`: This is the linear depth of the hit position in camera space. For example, `float z = (worldToView * hitPosition).z` yields the desired value.
- It also takes a couple of optional inputs that this sample does not provide for simplicity's sake.

A diagram showing how ViewZ is computed from the primary hit point and the camera plane.

These are its output textures:

- `OUT_DIFF_RADIANCE_HITDIST` - a texture containing the denoised diffuse image
- `OUT_SPEC_RADIANCE_HITDIST` - a texture containing the denoised specular image

The denoiser `RELAX_DIFFUSE_SPECULAR` takes the same inputs, which makes it
easier to switch between the two.

Consult NRDDescs.h to see which inputs are required. For each input type
("Resource Type") it also prescribes the required image format and other
dependencies.

NRD can only work, if it is provided with the right data. NRD's [README.md](https://github.com/NVIDIA-RTX/NRD/blob/master/README.md)
talks about this in much detail. Below we will mention a few critical points to
get right.

### Splitting diffuse and specular signals

While you could feed combined diffuse and specular lighting into a single
channel, denoising quality is much better when the two noisy signals are
separated at the primary hit (or PSR, as shown later). NRD therefore provides
dedicated diffuse+specular denoisers that take the two signals separately.

A diagram of multiple paths traced through a scene, showing separate
diffuse/specular paths, the "pathLength" from the first to the second hit, and
direct light sampling at the first hit.

Splitting the path traced image into a diffuse and specular image requires
separately following the diffuse and specular material's BSDF at the position
of the first non-mirrored hit. This sample deliberately picks the simplest
variant for clarity: it produces each signal at full resolution by tracing
**two** indirect paths per pixel - one along the diffuse BSDF and one along the
specular BSDF. Because both channels are always fully populated, the sample can
leave `HitDistanceReconstructionMode` disabled (`OFF`).

Be aware that this is the easy-to-read choice, **not** the performance-oriented
one. Tracing two indirect paths per pixel roughly doubles the indirect ray
cost. A production renderer typically follows only a **single** lobe per pixel
and traces **one** indirect path:

- a **probabilistic** approach: at the primary hit, randomly choose to follow
either the diffuse or the specular BSDF using a material-derived probability
(e.g. a Fresnel/luminance-based diffuse probability), and write the result
into the corresponding channel. Use Bayer or blue-noise dithering for that
choice so the selected lobes form a well-distributed spatial pattern.
- a **checkerboard** pattern, where every other pixel follows either a diffuse
or a specular path.

Both single-path variants leave each NRD input channel only partially populated
(one of the two signals is missing at any given pixel). The missing *radiance*
is not a problem on its own: write 0 for the missing signal, divide the
surviving samples by the lobe-selection probability so the mean is preserved,
and NRD's normal temporal+spatial passes average it back over the neighborhood.

The missing *hit distance*, however, does need explicit reconstruction. NRD
consumes hit distance early and per-pixel (for specular virtual-position /
reprojection, motion and disocclusion logic, and to derive the blur radius), so
a hole there would corrupt the specular tracking - and unlike radiance it can't
be recovered by averaging. This is exactly what `HitDistanceReconstructionMode`
is for: despite the broad-sounding name, it only reconstructs the *hit distance*
channel, filling each missing value from a valid neighbor in a 3x3 (`AREA_3X3`)
or 5x5 (`AREA_5X5`) window. To guarantee such a neighbor always exists, the
probabilistic choice must use Bayer/blue-noise dithering (not white noise) and
the diffuse probability must be clamped (to `[1/4, 3/4]` for 3x3). Because this
sample traces both signals every frame, hit distance is always valid and it
leaves `HitDistanceReconstructionMode` at `OFF`. Follow the NRD README.md file
for the exact requirements.

The secondary paths to compute the indirect radiance coming from the diffuse
BSDF's and specular BSDF's direction will then be performed in a regular
fashion (i.e. after the first bounce, both kinds of light path sample the full BSDF).
This is expressed in `shaders/nrd.slang` marked with `#DIFFUSE` and
`#SPECULAR`. The path tracing code for gathering the indirect light contributions
can be found in `pathtrace_rchit.slang`.

### Hit distances

NRD asks for a scalar 'hit distance' to be provided. This is typically the
path length along the ray between the primary hit (or PSR) and the next hit.
Hit distances can also be extended to accumulate along the path, but this
requires special handling (refer to NRD's [README.md](https://github.com/NVIDIA-RTX/NRD/blob/master/README.md)).
In this example, we chose the easier method of just using the secondary path
length.

Specific hit distances are used under these circumstances:

- if the ray is absorbed at the primary hit, we return a hit distance of 0.0.
- if the secondary ray hits the environment, we return a hit distance of `NRD_INF`.

### Mirror-like surfaces

Mirror-like surfaces work better with a special treatment. Instead of recording
the position and hit parameters at the surface of the mirror, we continue
following the mirrored ray until it hits something that is not a mirror - the
*Primary Surface Replacement (PSR)*. It looks like as if we can see the
mirrored objects "behind" the mirror as though the mirror acts as kind of a
portal into a virtual world. The G-Buffer records the data of the PSR at its
virtual world space, such as Normal, Roughness and ViewZ. This helps the denoiser
denoise reflected objects much better.

Look into `shaders/nrd.slang` for `#PSR` to find the shader code that implements
primary surface replacement and consult NRD SDK's README.md for more details.

### Demodulation

Demodulation is the act of removing *intended* 'noise' from the image before we
present it to the denoiser. These kinds of intended noise are typically details
coming from textures at the primary hit surface; for instance, grain in a wood texture. You certainly don't want the
denoiser recognizing these details as noise to remove. This would result in
blurry, less detailed surfaces after denoising. In addition, we completely
separate out noise-free signals like:

- light emissions of the primary surface;
- perfect reflections of the environment map;
- when the primary ray does not hit any geometry but instead hits the
environment map directly.

We follow NRD's README.md's suggestion for demodulating the diffuse and
specular signals in `nrd.slang`.

The `compositing.slang` shader will later multiply the denoised diffuse and
specular images by the inverse of the modulation factor and recombine all
parts of the image (specular, diffuse, direct light) into a single one.

### MIS weighting

The implemented path tracer uses importance sampling to estimate the
integral in the [rendering equation](https://en.wikipedia.org/wiki/Rendering_equation).
Care has to be taken when the
path tracer samples the same function multiple times, but uses different PDFs
when doing so. This sample is built such that, at the primary hit and each
segment of the indirect path, we sample the environment map for its direct
light contribution at that point. It then follows the material's BSDF to probe
for indirect lighting contribution at the hit point. Thus, for any surface
point along the path it can happen that the environment is sampled twice: once
via directly probing the environment map (the envmap has its own precomputed
sampling distribution) and once more when following the material's BSDF at each
hit point. Both contributions need to be weighted accordingly. In this sample
we chose the "power heuristic" to compute the MIS weights for both
contributions.

## Compositing

The final image will be composed of the denoised diffuse and specular color plus
direct lighting. Diffuse and specular are added only if there
was a hit and albedo is extracted from the base color and metalness information.
While the ray generation shader wrote out the "demodulated" diffuse and specular
colors to be denoised, the composition shader brings back the de-modulated 
detail into the final image.

## Authors and Metadata

Tags:

- raytracing, path-tracing, GLTF, HDR, tonemapper, picking, BLAS, TLAS, PBR
material, denoising, NRD

Extensions:

- VK_KHR_buffer_device_address, VK_KHR_acceleration_structure,VK_KHR_ray_tracing_pipeline, 
VK_KHR_ray_query, VK_KHR_push_descriptor, VK_KHR_shader_clock, VK_KHR_create_renderpass2

Authors:

- [Martin-Karl Lefrançois](https://developer.nvidia.com/blog/author/mlefrancois/)
- Mathias Heyer