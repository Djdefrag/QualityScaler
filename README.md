## QualityScaler
Image/video upscaling & enhancement GUI app based on BRSGAN &amp; RealSR_JPEG

![GUI](https://user-images.githubusercontent.com/32263112/163949184-c285734e-8be7-4b37-9f73-aa397f68eb19.png)

## Credits.

BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

RealSR_JPEG - https://github.com/jixiaozhong/RealSR | https://arxiv.org/pdf/2005.01996.pdf

## Installation.

QualityScaler is completely portable; just download, unzip and execute the file .exe

## Supported AI Backend.
* Nvidia cuda 10.2
   * compatible GPUs (including mobile version)
     * GTX 700/800/900/1000/1600/2000 
     * Tesla (up to T4)
     * Quadro (up to RTX T1000)
* CPU [works without GPU, but is very slow]

## Features.
* Easy to use GUI
* Images and video upscale
* Drag&drop files [image/multiple images/video]
* Automatic image tiling/merging to avoid Gpu VRam limitation
* Automatic defects resolution after tiling and merging image
* Different upscale factors:
  * x2   - upscale factor 2: 500x500px -> 1000x1000px
  * x3   - upscale factor 3: 500x500px -> 1500x1500px
  * x4   - upscale factor 4: 500x500px -> 2000x2000px
* Cpu and Gpu [cuda] backend
* Compatible images - PNG, JPEG, BMP, WEBP, TIF  
* Compatible video  - MP4, WEBM, GIF, MKV, FLV, AVI, MOV 

## Next steps.
* Support for Nvidia RTX 3k
* Use both model for the upscale
* Include audio for upscaled video
* Support for other GPUs (AMD, Intel) with new backend

## Example.

Original photo (200x200 px)
Upscaled photo x4 (800x800px)

![test bsrganx4](https://user-images.githubusercontent.com/32263112/163949737-627cc079-edcc-4abb-acd9-54b23a348012.png)


