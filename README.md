## QualityScaler
Image/video upscaling & enhancement GUI app based on BRSGAN &amp; RealSR_JPEG

![Immagine 2022-04-10 174252](https://user-images.githubusercontent.com/32263112/162680088-37c98fd5-08cd-458d-88c9-3c4cc1bcd3f3.jpg)


## Credits.

BSRGAN - https://github.com/cszn/BSRGAN | https://arxiv.org/abs/2103.14006

RealSR_JPEG - https://github.com/jixiaozhong/RealSR | https://arxiv.org/pdf/2005.01996.pdf

## Installation.

QualityScaler is completely portable; just download, unzip and execute the file .exe

## Supported AI Backend.
* Nvidia Cuda [10.2]
* CPU [works without GPU, but is very slow]

## Features.
* Easy to use GUI
* Images and video upscale
* Drag&drop files [image/multiple images/video]
* Different upscale factors:
  * auto - automatic choose best upscale factor for the GPU used (to avoid running out of VRAM)
  * x1   - mantain same resolution but reconstruct & enhance the image (ideal for bigger images) 
  * x2   - upscale factor 2: 500x500px -> 1000x1000px
  * x4   - upscale factor 4: 500x500px -> 2000x2000px
* Cpu and Gpu [cuda] backend
* Compatible images - PNG, JPEG, BMP, WEBP, TIF  
* Compatible video  - MP4, WEBM, GIF, MKV, FLV, AVI, MOV 

## Next steps.
* Use both model for the upscale
* Include audio for upscaled video
* Support for other GPUs (AMD, Intel) with new backend

## Example.

Original photo (200x200 px)

![220px-Lenna_test_image](https://user-images.githubusercontent.com/32263112/161437114-8ed041b2-e958-42df-9c7c-f71052d81948.png)

BSRGAN (800x800 px)

![220px-Lenna_test_image_resized_BSRGAN_x0](https://user-images.githubusercontent.com/32263112/161437168-2db8b791-e9be-45b8-bcdc-9ab49b3daa66.png)

RealSR_JPEG (800x800 px)

![220px-Lenna_test_image_resized_RealSR_JPEG_x0](https://user-images.githubusercontent.com/32263112/161437196-e1b81f58-5c71-41b7-b5ff-56ee215d885c.png)
