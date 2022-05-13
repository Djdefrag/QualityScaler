<div align="center">
    <img src="https://github.com/Djdefrag/QualityScaler/blob/main/logo.png" width="125"> </a> 
    <br><br> QualityScaler - Image-video upscaling/enhancement Windows app <br><br>
    <a href="https://github.com/Djdefrag/QualityScaler/releases">
         <img src="https://user-images.githubusercontent.com/86362423/162710522-c40c4f39-a6b9-48bc-84bc-1c6b78319f01.png" width="200">
    </a>
</div>
<br>
<div align="center">
    <img src="https://user-images.githubusercontent.com/32263112/163949184-c285734e-8be7-4b37-9f73-aa397f68eb19.png"> </a> 
</div>

## Credits.

BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

RealSR_JPEG - https://github.com/jixiaozhong/RealSR | https://arxiv.org/pdf/2005.01996.pdf

## Installation.

QualityScaler is completely portable:
1) download the release .zip
2) unzip using 7zip or similar
3) execute QualityScaler.exe in the directory

## Supported AI Backend.
* Nvidia cuda 11.1
   * compatible GPUs (including mobile version)
     * from GTX 800 to RTX 3000
     * Quadro compatible with cuda 11.1
     * Tesla compatible with cuda 11.1
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
* [v] Support for Nvidia RTX 3k
* Use both model for the upscale
* Include audio for upscaled video
* Support for other GPUs (AMD, Intel) with new backend

## Example.

![test](https://user-images.githubusercontent.com/32263112/166690007-f1601487-7b94-4f2c-b4e2-436bc189a26e.png)




