
<div align="center">
    <br>
    <img src="https://github.com/Djdefrag/QualityScaler/blob/main/Assets/logo.png" width="175"> </a> 
    <br><br> QualityScaler - image/video deeplearning upscaling for any GPU <br><br>
    <a href="https://github.com/Djdefrag/QualityScaler/releases">
         <img src="https://user-images.githubusercontent.com/86362423/162710522-c40c4f39-a6b9-48bc-84bc-1c6b78319f01.png" width="200">
    </a>
</div>
<br>
<div align="center">
    <img src="https://user-images.githubusercontent.com/32263112/200124160-5ca85aae-f09e-4e1f-a5cc-7e013b90ccf3.PNG"> </a> 
</div>


## Citations. ❤

https://80.lv/articles/80-level-digest-great-ai-powered-tools-for-upscaling-images/

https://timesavervfx.com/ai-upscale/

## No Watermarks ✨
https://jangystudio.itch.io/qualityscaler

## Credits.

BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

## How is made. 🛠

QualityScaler is completely written in Python, from backend to frontend. External packages are:
- [ ] AI  -> Pytorch-directml
- [ ] GUI -> Tkinter / Tkdnd / Sv_ttk
- [ ] Image/video -> OpenCV / Moviepy
- [ ] Packaging   -> Pyinstaller
- [ ] Miscellaneous -> Pywin32 / Win32mica / Image_slicer

## Installation. 👨‍💻
#### Prerequisites: 
 Visual C++: https://www.techpowerup.com/download/visual-c-redistributable-runtime-package-all-in-one/
 
 DirectX runtime: https://www.microsoft.com/en-us/download/details.aspx?id=8109
 
#### Installation:
 1. download the QualityScaler release .zip
 2. unzip using 7zip or similar
 3. execute QualityScaler.exe in the directory

## Requirements. 🤓
- [ ] Windows 11 / Windows 10
- [ ] RAM >= 8Gb
- [ ] Directx12 compatible GPU:
    - [ ] any AMD >= Radeon HD 7000 series
    - [ ] any Intel HD Integrated >= 4th-gen core
    - [ ] any NVIDIA >=  GTX 600 series
- [ ] CPU [works without GPU, but is very slow]

## My testing PC.
- [ ] Windows 10 ReviOS
- [ ] CPU Ryzen 5600G
- [ ] RAM 16Gb
- [ ] GPU Nvidia 1660
- [ ] STORAGE 1 Sata 120Gb SSD, 1 NVME 500Gb SSD

## Features.

- [x] Easy to use GUI
- [x] Image/list of images upscale
- [x] Video upscale
- [x] Drag&drop files [image/multiple images/video]
- [x] Automatic image tiling and merging to avoid gpu VRAM limitation
- [x] Resize image/video before upscaling
- [x] Cpu and Gpu backend
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫

- [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
- [x] New GUI with Windows 11 style
- [ ] Optimizing image/frame resize and frames extraction 
- [ ] Include audio for upscaled video
- [ ] Update libraries 
    - [ ] Python 3.10 (expecting ~10% more performance) 
    - [ ] Python 3.11 (expecting ~30% more performance)

## Known bugs.
- [x] Windows10 - the app starts with white colored navbar instead of dark
- [x] Upscaling multiple images doesn't free GPU Vram, so the it is very likely that the process will fail when the gpu memory fills up
- [ ] Filenames with non-latin symbols (for example kangy, cyrillic etc.) not supported - [Temp solution] rename files like "image" or "video"
- [ ] When running QualityScaler as Administrator, drag&drop is not working
- [ ] Some user reported that QualityScaler does not work correctly, returning the message 'Errore while upscaling'

## Some Example.

![test](https://user-images.githubusercontent.com/32263112/166690007-f1601487-7b94-4f2c-b4e2-436bc189a26e.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/168884625-c869baee-4cca-4a33-bdad-b65d9c29889d.png)

![Bsrgan x4 (2)](https://user-images.githubusercontent.com/32263112/197983965-40785dbd-78c6-48a0-a1eb-39d9c3278f42.png)

![Bsrgan x4 (3)](https://user-images.githubusercontent.com/32263112/197983979-5857a855-d402-4fab-9217-ee5bd057bd01.png)

![Bsrgan x4](https://user-images.githubusercontent.com/32263112/198290909-277e176e-ccb4-4a4b-8531-b182a18d566a.png)


