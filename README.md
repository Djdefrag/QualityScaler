<!DOCTYPE html>
<html>
<body>
    <div align="center">    
        <img src="https://github.com/Djdefrag/QualityScaler/blob/main/Assets/logo.png" width="175"> 
        <br><br> QualityScaler - image/video AI upscaler app <br><br>
        <a href="https://jangystudio.itch.io/qualityscaler">
            <button>
                <img src="https://static.itch.io/images/badge-color.svg" width="225" height="70">
            </button>     
        </a>
        <a href="https://store.steampowered.com/app/2463110/QualityScaler/">
            <button>
                 <img src="https://images.squarespace-cdn.com/content/v1/5b45fae8b98a78d9d80b9c5c/1531959264455-E7B8MJ3VMPX0593VGCZG/button-steam-available-fixed-2.png" width="250" height="70">
            </button>                 
        </a>
    </div>
    <br>
    <div align="center">
        <img src="https://github.com/Djdefrag/QualityScaler/assets/32263112/cdf45a13-579a-4f39-a64e-e60b80ac6bd9">
    </div>
</body>
</html>


## What is QualityScaler?
Qualityscaler is a Windows app powered by AI to enhance, enlarge and reduce noise in photographs and videos.

## Other AI projects.🤓
- https://github.com/Djdefrag/RealESRScaler / RealESRScaler - image/video AI upscaler app (Real-ESRGAN)
- https://github.com/Djdefrag/FluidFrames.RIFE / FluidFrames.RIFE - video AI interpolation app (RIFE-HDv3)


## Citations. ❤
- https://80.lv/articles/80-level-digest-great-ai-powered-tools-for-upscaling-images/
- https://timesavervfx.com/ai-upscale/

## Credits.
BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

## How is made. 🛠
QualityScaler is completely written in Python, from backend to frontend. 
External packages are:
- AI  -> torch / torch-directml
- GUI -> customtkinter
- Image/video -> openCV / moviepy
- Packaging   -> pyinstaller / upx

## Requirements. 🤓
- Windows 11 / Windows 10
- RAM >= 8Gb
- Any Directx12 compatible GPU with  >=  4GB VRAM

## Features.
- [x] Easy to use GUI
- [x] Images and Videos upscale
- [x] Multiple AI models
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Resize image/video before AI upscaling
- [x] Interpolation between the original and upscaled image/video
- [x] Multiple Gpu support
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫
- [ ] 1.X versions
    - [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
    - [x] New GUI with Windows 11 style
    - [x] Include audio for upscaled video
    - [x] Optimizing video frame resize and extraction speed
    - [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
    - [x] Python 3.10 (expecting ~10% more performance)
- [ ] 2.X versions (now under development)
    - [x] New, completely redesigned graphical interface based on @customtkinter
    - [x] Upscaling images and videos at once (currently it is possible to upscale images or single video)
    - [x] Upscale multiple videos at once
    - [x] Choose upscaled video extension
    - [x] Interpolation between the original and upscaled image/video
    - [ ] Python 3.11 (expecting ~30% more performance)
    - [ ] Torch/torch-directml 2.0 (expecting ~20% more performance)

## Some Example.

#### Videos
![original](https://user-images.githubusercontent.com/32263112/209139620-bdd028f8-d5fc-40de-8f3d-6b80a14f8aab.gif)

https://user-images.githubusercontent.com/32263112/209139639-2b123b83-ac6e-4681-b94a-954ed0aea78c.mp4

#### Images
![test](https://user-images.githubusercontent.com/32263112/166690007-f1601487-7b94-4f2c-b4e2-436bc189a26e.png)

![ORIGINAL](https://user-images.githubusercontent.com/32263112/226847190-e4dbda21-8896-456d-8120-3137f3d2ac62.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/168884625-c869baee-4cca-4a33-bdad-b65d9c29889d.png)

![Bsrgan x4 (2)](https://user-images.githubusercontent.com/32263112/197983965-40785dbd-78c6-48a0-a1eb-39d9c3278f42.png)

![Bsrgan x4 (3)](https://user-images.githubusercontent.com/32263112/197983979-5857a855-d402-4fab-9217-ee5bd057bd01.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/198290909-277e176e-ccb4-4a4b-8531-b182a18d566a.png)


