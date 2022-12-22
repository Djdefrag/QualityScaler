
<div align="center">
    <br>
    <img src="https://github.com/Djdefrag/QualityScaler/blob/main/Assets/logo.png" width="175"> </a> 
    <br><br> QualityScaler - image/video AI upscaler app (BSRGAN) <br><br>
    <a href="https://jangystudio.itch.io/qualityscaler">
         <img src="https://user-images.githubusercontent.com/86362423/162710522-c40c4f39-a6b9-48bc-84bc-1c6b78319f01.png" width="200">
    </a>
</div>
<br>
<div align="center">
    <img src="https://user-images.githubusercontent.com/32263112/208716221-83d7fde3-43fb-499d-8052-a1b0e950195c.PNG"> </a> 
</div>

## Other AI projects.🤓

https://github.com/Djdefrag/RealESRScaler / RealESRScaler - image/video AI upscaler app (Real-ESRGAN)


## Citations. ❤

https://80.lv/articles/80-level-digest-great-ai-powered-tools-for-upscaling-images/

https://timesavervfx.com/ai-upscale/

## Credits.

BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

## How is made. 🛠

QualityScaler is completely written in Python, from backend to frontend. External packages are:
- [ ] AI  -> Pytorch-directml
- [ ] GUI -> Tkinter / Tkdnd / Sv_ttk
- [ ] Image/video -> OpenCV / Moviepy
- [ ] Packaging   -> Pyinstaller
- [ ] Miscellaneous -> Pywin32 / Win32mica / split_image

## Requirements. 🤓
- [ ] Windows 11 / Windows 10
- [ ] RAM >= 8Gb
- [ ] Directx12 compatible GPU:
    - [ ] any AMD >= Radeon HD 7000 series
    - [ ] any Intel HD Integrated >= 4th-gen core
    - [ ] any NVIDIA >=  GTX 600 series

## Features.
- [x] Easy to use GUI
- [x] Image/list of images upscale
- [x] Video upscale
- [x] Drag&drop files [image / multiple images / video]
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Resize image/video before upscaling
- [x] Multiple Gpu support
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫

- [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
- [x] New GUI with Windows 11 style
- [x] Include audio for upscaled video
- [x] Optimizing video frame resize and extraction speed
- [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
- [ ] Update libraries 
    - [x] Python 3.10 (expecting ~10% more performance) 
    - [ ] Python 3.11 (expecting ~30% more performance)
    - [x] Pytorch-directml

## Known bugs.
- [x] Windows10 - the app starts with white colored navbar instead of dark
- [x] Upscaling multiple images doesn't free GPU Vram, so the it is very likely that the process will fail when the gpu memory fills up
- [ ] Filenames with non-latin symbols (for example kangy, cyrillic etc.) not supported - [Temp solution] rename files like "image" or "video"
- [ ] When running QualityScaler as Administrator, drag&drop is not working

## Some Example.

![original](https://user-images.githubusercontent.com/32263112/209139620-bdd028f8-d5fc-40de-8f3d-6b80a14f8aab.gif)

https://user-images.githubusercontent.com/32263112/209139639-2b123b83-ac6e-4681-b94a-954ed0aea78c.mp4

.

![test](https://user-images.githubusercontent.com/32263112/166690007-f1601487-7b94-4f2c-b4e2-436bc189a26e.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/168884625-c869baee-4cca-4a33-bdad-b65d9c29889d.png)

![Bsrgan x4 (2)](https://user-images.githubusercontent.com/32263112/197983965-40785dbd-78c6-48a0-a1eb-39d9c3278f42.png)

![Bsrgan x4 (3)](https://user-images.githubusercontent.com/32263112/197983979-5857a855-d402-4fab-9217-ee5bd057bd01.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/198290909-277e176e-ccb4-4a4b-8531-b182a18d566a.png)


