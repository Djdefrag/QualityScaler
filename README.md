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
        <img src="https://github.com/Djdefrag/QualityScaler/assets/32263112/57e90bfa-52ac-42e3-bc73-e72690a75697">
    </div>
</body>
</html>


## What is QualityScaler?
Qualityscaler is a Windows app powered by AI to enhance, upscale and de-noise photographs and videos.

## Installation
1. Download the [latest release](../../releases/latest) or clone the repository
   ```sh
   git clone https://github.com/Djdefrag/QualityScaler.git
   ```
2. Install requirements (using Python > 3, &le; 3.10 )
   ```sh
   cd QualityScaler
   python -m pip install --requirement ./requirements.txt  # torch-directml requires Python 3.10
   ```
3. Start QualityScaler
   ```sh
   python ./QualityScaler.py
   ```

## Other AI projects.🤓
- https://github.com/Djdefrag/RealScaler / RealScaler - image/video AI upscaler (Real-ESRGAN)
- https://github.com/Djdefrag/FluidFrames.RIFE / FluidFrames.RIFE - video AI frame generation

## Credits.
- BSRGAN - https://github.com/cszn/BSRGAN
- Real-ESRGAN - https://github.com/xinntao/Real-ESRGAN
- IRCNN - https://github.com/lipengFu/IRCNN

## Citations. ❤
- https://80.lv/articles/80-level-digest-great-ai-powered-tools-for-upscaling-images/
- https://timesavervfx.com/ai-upscale/

## How is made. 🛠
QualityScaler is completely written in Python, from backend to frontend. 
External packages are:
- AI  -> torch / onnxruntime-directml
- GUI -> customtkinter
- Image/video -> OpenCV / moviepy
- Packaging   -> Pyinstaller

## Make it work by yourself. 👨‍💻
Prerequisites.
- Python installed on your pc, you can download it from here (https://www.python.org/downloads/release/python-3119/)
- VSCode installed on your pc, you can download it from here (https://code.visualstudio.com/)

Getting started.
- First of all, you need to download the project on your PC (Green button Code > Download ZIP)
- Extract the project directory from the .zip
- Now you need to download the AI models (github won't let me upload them directly because they are too big)
- In "AI-onnx" folder, there is the link to download the AI models, download the .zip and extract the files in AI-onnx directory
- Open the project with VSCode (just Drag&Drop the project directory on VSCode)
- Click on QualityScaler.py from left bar (VSCode will ask you to install some plugins, go ahead)
- Now, you need to install dependencies. In VSCode there is the "Terminal" panel, click there and execute the command "pip install -r requirements"
- Close VSCode and re-open it (this will refresh all the dependecies installed)
- Just click on the "Play button" in the upper right corner of VSCode
- Now the app should work

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
- [x] Compatible images - png, jpeg, bmp, webp, tif  
- [x] Compatible video  - mp4, wemb, gif, mkv, flv, avi, mov, qt 

## Next steps. 🤫
- [x] 1.X versions
    - [x] Switch to Pytorch-directml to support all Directx12 compatible gpu (AMD, Intel, Nvidia)
    - [x] New GUI with Windows 11 style
    - [x] Include audio for upscaled video
    - [x] Optimizing video frame resize and extraction speed
    - [x] Multi GPU support (for pc with double GPU, integrated + dedicated)
    - [x] Python 3.10 (expecting ~10% more performance)
- [x] 2.X versions
    - [x] New, completely redesigned graphical interface based on @customtkinter
    - [x] Upscaling images and videos at once (currently it is possible to upscale images or single video)
    - [x] Upscale multiple videos at once
    - [x] Choose upscaled video extension
    - [x] Interpolation between the original and upscaled image/video
    - [x] More Interpolation levels (Low, Medium, High)
    - [x] Show the remaining time to complete video upscaling
    - [x] Support for SRVGGNetCompact AI architecture
    - [x] Metadata extraction and application from original file to upscaled file (via exiftool)
    - [x] Support for SAFMN AI architecture
- [ ] 3.X versions
    - [x] New AI engine powered by onnxruntime-directml (https://pypi.org/project/onnxruntime-directml/)
    - [x] Python 3.11 (~10% performance improvements)
    - [x] Display images/videos upscaled resolution in the GUI
    - [x] FFMPEG 7 (latest release)
    - [x] Video multi-threading AI upscale 
    - [x] Python 3.12
    - [ ] Video upscaling pause and restart

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


