
<div align="center">
    <br>
    <img src="https://github.com/Djdefrag/QualityScaler/blob/main/logo.png" width="125"> </a> 
    <br><br> QualityScaler - Image and video upscaling/enhancement Windows app <br><br>
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
- [x] Easy to use GUI
- [x] Image/list of images upscale
- [x] Video upscale
- [x] Drag&drop files [image/multiple images/video]
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Different upscale factors:
    - [x] x2   - 500x500px -> 1000x1000px
    - [x] x3   - 500x500px -> 1500x1500px
    - [x] x4   - 500x500px -> 2000x2000px
- [x] Cpu and Cuda backend
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps.
- [x] Support for Nvidia RTX 3k and 2k with cuda 11
- [ ] New Fused model (that combines the best of both models)
- [ ] New GUI with Windows 11 style
- [ ] Include audio for upscaled video
- [ ] Switch to Pytorch-directml to support all existing gpu (AMD, Intel, Nvidia)
- [ ] Update libraries 
    - [ ] Python 3.10 (expecting ~10% more performance) 
    - [ ] Python 3.11 (expecting ~30% more performance, now in beta)

## Example.

![test](https://user-images.githubusercontent.com/32263112/166690007-f1601487-7b94-4f2c-b4e2-436bc189a26e.png)


