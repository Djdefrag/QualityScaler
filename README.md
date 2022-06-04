
<div align="center">
    <br>
    <img src="https://github.com/Djdefrag/QualityScaler/blob/main/logo.png" width="175"> </a> 
    <br><br> Image/video deepLearning upscaler app for Windows <br><br>
    <a href="https://github.com/Djdefrag/QualityScaler/releases">
         <img src="https://user-images.githubusercontent.com/86362423/162710522-c40c4f39-a6b9-48bc-84bc-1c6b78319f01.png" width="200">
    </a>
</div>
<br>
<div align="center">
    <img src="https://user-images.githubusercontent.com/32263112/171654121-06709d1c-0551-429a-9467-fd4618019b7e.png"> </a> 
</div>

## Credits.

BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

RealSR_JPEG - https://github.com/jixiaozhong/RealSR | https://arxiv.org/pdf/2005.01996.pdf

## How is made.

QualityScaler is completely written in Python, from backend to frontend. External packages are:
- [ ] AI  -> Pytorch-directml
- [ ] GUI -> Tkinter / Tkdnd / Sv_ttk
- [ ] Image/video -> OpenCV / Moviepy
- [ ] Packaging   -> Pyinstaller
- [ ] Miscellaneous -> Pywin32 / Win32mica / Image_slicer

## Installation.

1) download the QualityScaler release .zip
2) unzip using 7zip or similar
3) execute QualityScaler.exe in the directory

## Requirements
- [ ] Windows 11 / Windows 10
- [ ] RAM >= 8Gb
- [ ] Directx12 compatible GPU:
    - [ ] any AMD >= Radeon HD 7000 series
    - [ ] any Intel HD Integrated >= 4th-gen core
    - [ ] any NVIDIA >=  GTX 600 series
- [ ] CPU [works without GPU, but is very slow]

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
- [x] New Fused model (that combines the best of both models)
- [x] New GUI with Windows 11 style
- [ ] Include audio for upscaled video
- [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
- [ ] Convert to Cython core functions
    - [ ] expecting ~50% more performance
- [ ] Update libraries 
    - [ ] Python 3.10 (expecting ~10% more performance) 
    - [ ] Python 3.11 (expecting ~30% more performance, now in beta)

## Some Example.

![test](https://user-images.githubusercontent.com/32263112/166690007-f1601487-7b94-4f2c-b4e2-436bc189a26e.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/168884625-c869baee-4cca-4a33-bdad-b65d9c29889d.png)

![ggg](https://user-images.githubusercontent.com/32263112/168884634-fc3fdc7b-ac77-4750-aaf6-54b16786dacf.png)

![example](https://user-images.githubusercontent.com/32263112/171657072-0a746274-46e9-4641-b16c-a9f6f612624b.png)




