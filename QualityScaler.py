
import base64
import ctypes
import functools
import multiprocessing
import os
import os.path
import shutil
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkFont
import webbrowser
import zlib
from pathlib import Path
from timeit import default_timer as timer
from tkinter import PhotoImage, ttk

import cv2
import moviepy.video.io.ImageSequenceClip
import numpy as np
import tkinterDnD
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from image_slicer import join
from image_slicer import slice as img_cutter
from PIL import Image

ctypes.windll.shcore.SetProcessDpiAwareness(True)
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

if scaleFactor == 1.0:
    font_scale = 1.25
elif scaleFactor == 1.25:
    font_scale = 1.0
else:
    font_scale = 0.85

version    = "1.4.0 - stable"

author     = "Annunziata Gianluca"
paypalme   = "https://www.paypal.com/paypalme/jjstd/5"
githubme   = "https://github.com/Djdefrag"
patreonme  = "https://www.patreon.com/Djdefrag"

image_path          = "no file"
AI_model            = "BSRGAN" 
device              = "cuda"
upscale_factor      = 2
cut_image_factor    = 170
actual_step         = ""
single_file         = False
multiple_files      = False
video_files         = False
multi_img_list             = []
video_frames_list          = []
video_frames_upscaled_list = []
original_video_path = ""

supported_file_list = [ '.jpg' , '.jpeg', '.JPG', '.JPEG',
                        '.png' , '.PNG' ,
                        '.webp', '.WEBP',
                        '.bmp' , '.BMP',
                        '.tif' , '.tiff', '.TIF', '.TIFF',
                        '.mp4', '.MP4', 
                        '.webm', '.WEBM',
                        '.mkv', '.MKV',
                        '.flv', '.FLV',
                        '.gif', '.GIF',
                        '.m4v', ',M4V',
                        '.avi', '.AVI',
                        '.mov', '.MOV',
                         '.3gp', '.mpg', '.mpeg'] 

supported_video_list = ['.mp4', '.MP4', 
                        '.webm', '.WEBM',
                        '.mkv', '.MKV',
                        '.flv', '.FLV',
                        '.gif', '.GIF',
                        '.m4v', ',M4V',
                        '.avi', '.AVI',
                        '.mov', '.MOV',
                         '.3gp', '.mpg', '.mpeg']

not_supported_file_list = ['.txt', '.exe', '.xls', '.xlsx', '.pdf',
                           '.odt', '.html', '.htm', '.doc', '.docx',
                           '.ods', '.ppt', '.pptx', '.aiff', '.aif',
                           '.au', '.bat', '.java', '.class',
                           '.csv', '.cvs', '.dbf', '.dif', '.eps',
                           '.fm3', '.psd', '.psp', '.qxd',
                           '.ra', '.rtf', '.sit', '.tar', '.zip',
                           '.7zip', '.wav', '.mp3', '.rar', '.aac',
                           '.adt', '.adts', '.bin', '.dll', '.dot',
                           '.eml', '.iso', '.jar', '.py',
                           '.m4a', '.msi', '.ini', '.pps', '.potx',
                           '.ppam', '.ppsx', '.pptm', '.pst', '.pub',
                           '.sys', '.tmp', '.xlt']


# ---------------------- Dimensions ----------------------

window_width       = 1300
window_height      = 725
left_bar_width     = 420
left_bar_height    = window_height
drag_drop_width    = window_width - left_bar_width
drag_drop_height   = window_height
button_width       = 250
button_height      = 34
show_image_width   = drag_drop_width * 0.9
show_image_height  = drag_drop_width * 0.7
image_text_width   = drag_drop_width * 0.9
image_text_height  = 34
button_1_y = 120
button_2_y = 170
button_3_y = 220
drag_drop_background = "#303030"
drag_drop_text_color = "#858585"
default_font         = 'Calibri'

# ---------------------- /Dimensions ----------------------

# ---------------------- Functions ----------------------

# ------------------ Neural Net related ------------------

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

# ------------------ /Neural Net related ------------------

# ------------------------ Utils ------------------------

def openpaypal():
    webbrowser.open(paypalme, new=1)

def opengithub():
    webbrowser.open(githubme, new=1)

def openpatreon():
    webbrowser.open(patreonme, new=1)

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def create_temp_dir(name_dir):
    # first delete the folder if exists
    if os.path.exists(name_dir):
        shutil.rmtree(name_dir)

    # then create a new folder
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

def find_file_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def imread_uint(path, n_channels=3):
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def create_void_logo():
    ICON = zlib.decompress(base64.b64decode('eJxjYGAEQgEBBiDJwZDBy'
        'sAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc='))
    _ , ICON_PATH = tempfile.mkstemp()
    with open(ICON_PATH, 'wb') as icon_file:
        icon_file.write(ICON)
    return ICON_PATH

def slice_image(img_file, num_tiles):
    tiles = img_cutter(img_file, num_tiles)
    return tiles

def reunion_image(tiles):
    image = join(tiles)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def delete_tiles_from_disk(tiles):
    #remove tiles file
    for tile in tiles:
        os.remove(tile.filename)

def resize_image_to_show(image_to_prepare):
    max_photo_resolution_value = show_image_height
    
    old_image     = cv2.imread(image_to_prepare)
    actual_width  = old_image.shape[1]
    actual_height = old_image.shape[0]

    max_val = max(actual_width, actual_height)

    if max_val >= max_photo_resolution_value:
        downscale_factor = max_val/max_photo_resolution_value
        new_width      = round(old_image.shape[1]/downscale_factor)
        new_height     = round(old_image.shape[0]/downscale_factor)
        resized_image  = cv2.resize(old_image,
                                (new_width, new_height),
                                interpolation = cv2.INTER_LINEAR)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"
    else:
        return image_to_prepare

# ----------------------- /Utils ------------------------

def optimize_torch(device):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    if 'cpu' in device:
        torch.set_num_threads(4)
    elif 'cuda' in device:
        torch.backends.cudnn.benchmark = True

def prepare_torch_model(AI_model, device):
    if 'cpu' in device:
        backend = torch.device('cpu')
    elif 'cuda' in device:
        backend = torch.device('cuda')
    
    model_path = find_file_by_relative_path(AI_model + ".pth")
    
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _ , v in model.named_parameters():
        v.requires_grad = False
    
    # not working, maibe torch > 1.8.2 ?
    # model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
    
    return model.to(backend, non_blocking=True)

def torch_AI_upscale_multiple_images(image_list, AI_model, upscale_factor, device):
    try:
        # 0) define the model
        model = prepare_torch_model(AI_model, device)
        optimize_torch(device)

        # 1) resize all images
        downscaled_images = []
        for image in image_list:
            img_downscaled = resize_image(image, upscale_factor)
            downscaled_images.append(img_downscaled)

        
        for img in downscaled_images:
            result_path = (img.replace("_resized.png","").replace(".png","") + 
                            "_"  + AI_model +
                            "_x" + str(upscale_factor) + 
                            ".png")

            # 2) calculating best slice number
            img_tmp = cv2.imread(img)
            val = min(img_tmp.shape[1], img_tmp.shape[0])
            num_tiles = round(val/cut_image_factor)
            if (num_tiles % 2) != 0:
                num_tiles += 1

            # 3) divide the image in tiles
            tiles = slice_image(img, num_tiles)

            # 4) upscale each tiles
            with torch.no_grad():
                for tile in tiles:
                    tile_adapted  = adapt_image_for_deeplearning(tile.filename)
                    tile_adapted  = tile_adapted.to(device, non_blocking = True)
                    tile_upscaled = model(tile_adapted)
                    tile_upscaled = tensor2uint(tile_upscaled)
                    imsave(tile_upscaled, tile.filename)
                    tile.image  = Image.open(tile.filename)
                    tile.coords = (tile.coords[0]*4, tile.coords[1]*4)

            # 5) then reconstruct the image by tiles upscaled
            image_upscaled = reunion_image(tiles)
        
            # 6) remove tiles file
            delete_tiles_from_disk(tiles)

            # 7) save reconstructed image
            cv2.imwrite(result_path, image_upscaled)
    except:
        error_root = tkinterDnD.Tk()
        ErrorMessage(error_root,  "upscale_problem")

def torch_AI_upscale_video_frames(video_frames_list, AI_model, upscale_factor, device):
    try:
        # 0) define the model
        model = prepare_torch_model(AI_model, device)
        optimize_torch(device)

        # 1) resize all images
        downscaled_images = []
        for image in video_frames_list:
            img_downscaled = resize_image(image, upscale_factor)
            downscaled_images.append(img_downscaled)

        for img in downscaled_images:
            result_path = (img.replace("_resized.png","").replace(".png","") + 
                            "_"  + AI_model +
                            "_x" + str(upscale_factor) + 
                            ".png")

            # 2) calculating best slice number
            img_tmp = cv2.imread(img)
            val = min(img_tmp.shape[1], img_tmp.shape[0])
            num_tiles = round(val/cut_image_factor)
            if (num_tiles % 2) != 0:
                num_tiles += 1

            # 3) divide the image in tiles
            tiles = slice_image(img, num_tiles)

            # 4) upscale each tiles
            with torch.no_grad():
                for tile in tiles:
                    tile_adapted  = adapt_image_for_deeplearning(tile.filename)
                    tile_adapted  = tile_adapted.to(device, non_blocking = True)
                    tile_upscaled = model(tile_adapted)
                    tile_upscaled = tensor2uint(tile_upscaled)
                    imsave(tile_upscaled, tile.filename)
                    tile.image  = Image.open(tile.filename)
                    tile.coords = (tile.coords[0]*4, tile.coords[1]*4)

            # 5) then reconstruct the image by tiles upscaled
            image_upscaled = reunion_image(tiles)
            
            # 6) remove tiles file
            delete_tiles_from_disk(tiles)

            # 7) save reconstructed image
            cv2.imwrite(result_path, image_upscaled)
    except:
        error_root = tkinterDnD.Tk()
        ErrorMessage(error_root, "upscale_problem")
    
def function_drop(event):
    global image_path
    global multiple_files
    global multi_img_list
    global video_files
    global single_file
    
    info_string.set("")

    supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number = count_files_dropped(event)
    all_supported, single_file, multiple_files, video_files, more_than_one_video = check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number)

    if video_files:
        # video section
        if not all_supported:
            info_string.set("Some files are not supported")
            return
        elif all_supported:
            if multiple_files:
                info_string.set("Only one video supported")
                return
            elif not multiple_files:
                if not more_than_one_video:
                    file_vid = str(event.data).replace("{", "").replace("}", "")
                    show_video_info_with_drag_drop(file_vid)
                    thread_extract_frames = threading.Thread(target = extract_frames_from_video, 
                                                            args    = (file_vid, 1), 
                                                            daemon  = True)
                    thread_extract_frames.start()

                    # reset variable
                    image_path     = "no file"
                    multi_img_list = []

                elif more_than_one_video:
                    info_string.set("Only one video supported")
                    return
    else:
        # image section
        if not all_supported:
            if multiple_files:
                info_string.set("Some files are not supported")
                return
            elif single_file:
                info_string.set("This file is not supported")
                return
        elif all_supported:
            if multiple_files:
                image_list_dropped = from_string_to_image_list(event)

                #convert images to png
                thread_convert_images = threading.Thread(target  = convert_multi_images_to_png, 
                                                         args    = (image_list_dropped, 1), 
                                                         daemon  = True)
                thread_convert_images.start()

                # convert only strings to .png 
                # of dropped files

                show_list_images_in_GUI_with_drag_drop(image_list_dropped)
                multi_img_list = convert_only_image_filenames(image_list_dropped)

                # reset variable
                image_path        = "no file"
                video_frames_list = []
            
            elif single_file:
                image_list_dropped             = from_string_to_image_list(event)

                #convert images to png
                thread_convert_images          = threading.Thread(target = convert_single_image_to_png_and_save_and_show_in_GUI, 
                                                                args   = (str(image_list_dropped[0]), 1), 
                                                                daemon  = True)
                thread_convert_images.start()

                multi_img_list = convert_only_image_filenames(image_list_dropped)

                # reset variable
                image_path        = "no file"
                video_frames_list = []

def upscale_button_command():
    global image_path
    global multiple_files
    global actual_step
    global process_upscale
    global upscale_factor
    global video_frames_list
    global video_files
    global video_frames_upscaled_list
    global original_video_path
    global device

    if "no model" in AI_model:
        info_string.set("No AI model selected!")
        return
    
    if video_files:
        if "extracting" in actual_step:
            info_string.set("Waiting for frames extraction!")
            return
        elif "ready" in actual_step:
            info_string.set("Upscaling video")
            place_stop_button()
            
            process_upscale = multiprocessing.Process(target = torch_AI_upscale_video_frames, 
                                                      args   = (video_frames_list, AI_model, upscale_factor, device))
            process_upscale.start()

            thread_wait = threading.Thread(target  = thread_wait_for_videoframes_and_video_reconstruct, 
                                            args   = (video_frames_list, AI_model, upscale_factor, original_video_path),
                                            daemon = True)
            thread_wait.start()

    elif multiple_files:
        if "converting" in actual_step:
            info_string.set("Waiting for images conversion!")
            return
        elif "ready"    in actual_step:
            place_stop_button()
            info_string.set("Upscaling multiple images")
            
            process_upscale = multiprocessing.Process(target = torch_AI_upscale_multiple_images, 
                                                      args   = (multi_img_list, AI_model, upscale_factor, device))
            process_upscale.start()

            thread_wait = threading.Thread(target = thread_wait_for_multiple_file, 
                                           args   = (multi_img_list, AI_model, upscale_factor),
                                           daemon = True)
            thread_wait.start()
    
    elif single_file:
        place_stop_button()
        info_string.set("Upscaling single image")

        process_upscale = multiprocessing.Process(target = torch_AI_upscale_multiple_images, 
                                                    args   = (multi_img_list, AI_model, upscale_factor, device))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_wait_for_multiple_file, 
                                           args   = (multi_img_list, AI_model, upscale_factor),
                                           daemon = True)
        thread_wait.start()
    
    elif "no file" in image_path:
        info_string.set("No file selected!")

def stop_button_command():
    global process_upscale
    process_upscale.terminate()
    info_string.set("Stopped")
    place_upscale_button()

def resize_image(image_to_prepare, upscale_factor):
    image_to_prepare = image_to_prepare.replace("{", "").replace("}", "")
    new_image_path = image_to_prepare
    
    resize_algorithm = cv2.INTER_AREA

    if upscale_factor == 2:
        # upscale x2
        # resize image dividing by 2

        old_image      = cv2.imread(image_to_prepare)
        new_width      = round(old_image.shape[1]/2)
        new_height     = round(old_image.shape[0]/2)
        resized_image  = cv2.resize(old_image,
                                    (new_width, new_height),
                                    interpolation = resize_algorithm)
        new_image_path = new_image_path.replace(".png", "_resized.png")
        cv2.imwrite(new_image_path, resized_image)
        return new_image_path
    
    elif upscale_factor == 3:
        # upscale x3
        # resize image subtracting 1/4 in resolution
        # ex. 1000 --> 750 --> 3000
        
        old_image      = cv2.imread(image_to_prepare)
        
        width_to_remove = round(old_image.shape[1]/4)
        height_to_remove = round(old_image.shape[0]/4)

        new_width      = round(old_image.shape[1] - width_to_remove)
        new_height     = round(old_image.shape[0] - height_to_remove)
        resized_image  = cv2.resize(old_image,
                                    (new_width, new_height),
                                    interpolation = resize_algorithm)
        new_image_path = new_image_path.replace(".png", "_resized.png")
        cv2.imwrite(new_image_path, resized_image)
        return new_image_path
    
    elif upscale_factor == 4:
        # upscale by 4
        # so simply return the image as it is

        return new_image_path

def extract_frames_from_video(video_path, _ ):
    global actual_step
    global video_frames_list
    global original_video_path

    create_temp_dir("QualityScaler_temp")
    original_video_path = video_path

    actual_step = "extracting" 
    info_string.set("Extracting frames from video")
    
    cap               = cv2.VideoCapture(video_path)
    num_frame         = 0
    video_frames_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        num_frame += 1
        result_path = "QualityScaler_temp" + os.sep + "frame_" + str(num_frame) + ".png"
        cv2.imwrite(result_path, frame)
        video_frames_list.append(result_path)
        info_string.set("Extracting frames n. " + str(num_frame))
    
    cap.release()
    actual_step = "ready"
    info_string.set("")

def video_reconstruction_by_frames(original_video_path, video_frames_upscaled_list, AI_model, upscale_factor):
    info_string.set("Processing video")
    
    # 1) get original video informations
    cap          = cv2.VideoCapture(original_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    path_as_list = original_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = original_video_path.replace(video_name, "")
    cap.release()

    # 2) remove any file extension from original video path string
    for video_type in supported_video_list:
        video_name = video_name.replace(video_type, "")

    # 3) create upscaled video path string
    upscaled_video_name = (only_path + 
                            video_name + 
                            "_" + 
                            AI_model + 
                            "_x" + 
                            str(upscale_factor) + 
                            ".mp4")

    # 4) create upscaled video with upscaled frames 
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(video_frames_upscaled_list, fps = frame_rate)
    clip.write_videofile(upscaled_video_name)

def check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number):
    all_supported                     = True
    single_file                       = False
    multiple_files                    = False
    video_files                       = False
    more_than_one_video               = False
    
    if not_supported_file_dropped_number > 0:
        all_supported = False
        
    if supported_file_dropped_number + not_supported_file_dropped_number == 1:
        single_file = True
    elif supported_file_dropped_number + not_supported_file_dropped_number > 1:
        multiple_files = True

    if supported_video_dropped_number == 1:
        video_files         = True     
        more_than_one_video = False
    elif supported_video_dropped_number > 1:
        video_files         = True
        more_than_one_video = True

    return all_supported, single_file, multiple_files, video_files, more_than_one_video

def count_files_dropped(event):
    supported_file_dropped_number     = 0
    not_supported_file_dropped_number = 0
    supported_video_dropped_number    = 0

    # count compatible images files
    for file_type in supported_file_list:
        supported_file_dropped_number = supported_file_dropped_number + str(event.data).count(file_type)
    
    # count compatible video files
    for file_type in supported_video_list:
        supported_video_dropped_number = supported_video_dropped_number + str(event.data).count(file_type)
    
    # count not supported files
    for file_type in not_supported_file_list:
        not_supported_file_dropped_number = not_supported_file_dropped_number + str(event.data).count(file_type)

    return supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number

def adapt_image_for_deeplearning(img):
    img = imread_uint(img, n_channels=3)
    img = uint2tensor4(img)
    return img
    
def convert_frames_list_uspcaled(video_frames_list, AI_model, upscale_factor):
    video_frames_upscaled_list = []
    for image in video_frames_list:
        result_path = image.replace(".png","") + "_" + AI_model + "_x" + str(upscale_factor) + ".png"
        video_frames_upscaled_list.append(result_path)

    return video_frames_upscaled_list   

def images_to_check_convert_filenames(image_list, AI_model, upscale_factor):
    temp_images_to_delete = []
    image_list_to_check = []
    
    for image in image_list:
        temp = image.replace(".png","") + "_" + AI_model + "_x" + str(upscale_factor) + ".png"
        image_list_to_check.append(temp)
        temp_images_to_delete.append(image.replace(".png","_resized.png"))

    return image_list_to_check, temp_images_to_delete

def thread_wait_for_multiple_file(image_list, AI_model, upscale_factor):
    start     = timer()
    image_list_to_check, temp_images_to_delete = images_to_check_convert_filenames(image_list,
                                                                                     AI_model, 
                                                                                     upscale_factor)
    # check if files exist
    how_many_images = len(image_list_to_check)
    counter_done    = 0
    for image in image_list_to_check:
        while not os.path.exists(image):
            time.sleep(1)

        if os.path.isfile(image):
            counter_done += 1
            info_string.set("Upscaled images " + str(counter_done) + "/" + str(how_many_images))

        if counter_done == how_many_images:  
            # delete temp files
            if len(temp_images_to_delete) > 0:
                for to_delete in temp_images_to_delete:
                    if os.path.exists(to_delete):
                        os.remove(to_delete)  

            end       = timer()
            info_string.set("Upscale completed [" + str(round(end - start)) + " sec.]")
            place_upscale_button()

def thread_wait_for_videoframes_and_video_reconstruct(video_frames_list, AI_model, upscale_factor, original_video_path):
    start = timer()
    counter_done = 0
    image_list_to_check, _ = images_to_check_convert_filenames(video_frames_list,
                                                               AI_model, 
                                                               upscale_factor)
    how_many_images = len(image_list_to_check)

    for image in image_list_to_check:
        while not os.path.exists(image):
            time.sleep(1)

        if os.path.isfile(image):
            counter_done += 1
            info_string.set("Upscaled frames " + str(counter_done) + "/" + str(how_many_images))

            if counter_done == how_many_images:  
                video_reconstruction_by_frames(original_video_path, 
                                            image_list_to_check,
                                            AI_model, 
                                            upscale_factor)
                
                end = timer()
                info_string.set("Video upscale completed [" + str(round(end - start)) + " sec.]")
                place_upscale_button()

def from_string_to_image_list(event):
    image_list = str(event.data).replace("{", "").replace("}", "")
    
    for file_type in supported_file_list:
        image_list = image_list.replace(file_type, file_type+"\n")

    image_list = image_list.split("\n")
    image_list.pop() # to remove last void element

    return image_list

def convert_only_image_filenames(image_list):
    list_converted = []
    for image in image_list:
        image = image.strip().replace("{", "").replace("}", "")
        for file_type in supported_file_list:
            image = image.replace(file_type,".png")
        
        list_converted.append(image)

    return list(dict.fromkeys(list_converted))

def convert_multi_images_to_png(image_list, _ ):
    global actual_step

    actual_step    = "converting"
    info_string.set("Converting images...")
    
    for image in image_list:
        image = image.strip()
        convert_single_image_to_png_and_save(image)
    
    actual_step    = "ready"
    info_string.set("")

def convert_single_image_to_png_and_save(image_to_prepare):
    image_to_prepare = image_to_prepare.replace("{", "").replace("}", "")
    if ".png" in image_to_prepare:
        return image_to_prepare
    else:
        new_image_path = image_to_prepare
        for file_type in supported_file_list:
            new_image_path = new_image_path.replace(file_type,".png")
        
        image_to_convert      = cv2.imread(image_to_prepare)
        cv2.imwrite(new_image_path, image_to_convert)
        return new_image_path

def convert_single_image_to_png_and_save_and_show_in_GUI(image_to_prepare, _ ):
    image_to_prepare = image_to_prepare.replace("{", "").replace("}", "")
    if ".png" in image_to_prepare:
        show_image_in_GUI_with_drag_drop(image_to_prepare)
    else:
        new_image_path = image_to_prepare
        for file_type in supported_file_list:
            new_image_path = new_image_path.replace(file_type,".png")
        
        image_to_convert      = cv2.imread(image_to_prepare)
        cv2.imwrite(new_image_path, image_to_convert)
        show_image_in_GUI_with_drag_drop(new_image_path)

# ---------------------- GUI related ---------------------- 

def clear_drag_drop_background():
    drag_drop = ttk.Label(root,
                          ondrop     = function_drop,
                          relief     = "flat",
                          background = drag_drop_background,
                          foreground = drag_drop_text_color)
    drag_drop.place(x=left_bar_width, y=0, width = drag_drop_width, height = drag_drop_height)

def show_video_info_with_drag_drop(video_path):
    clear_drag_drop_background()

    cap        = cv2.VideoCapture(video_path)
    width      = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height     = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    file_size  = Path(video_path).stat().st_size
    duration   = num_frames/frame_rate
    minutes    = int(duration/60)
    seconds    = duration%60
    path_as_list = video_path.split("/")
    video_name = str(path_as_list[-1])
    cap.release()

    file_description = ("\n" 
                        + " Path: "     + video_path.replace(video_name, "") + "\n\n" 
                        + " File: "     + video_name                         + "\n\n"
                        + " Resolution: " + str(width) + "x" + str(height)   + "\n\n"
                        + " Size: "      + str(truncate(file_size / 1048576, 2))  + " MB" + "\n\n"
                        + " Duration: "  + str(minutes) + ':' + str(round(seconds))       + "\n\n"
                        + " Frames: "    + str(num_frames)        + "\n\n"
                        + " Fps: "       + str(round(frame_rate)) + "\n\n")
    
    video_header = ttk.Label(root,
                            text       = "Video info",
                            ondrop     = function_drop,
                            font       = (default_font, round(12 * font_scale), "bold"),
                            anchor     = "center",
                            relief     = "flat",
                            justify    = "center",
                            background = "#181818",
                            foreground = "#D3D3D3")
    video_header.place(x = left_bar_width + drag_drop_width/2 - 750/2,
                       y = drag_drop_height/2 - 400/2 - 45,
                       width  = 200,
                       height = 35)

    video_info_space = ttk.Label(root,
                            text       = file_description,
                            ondrop     = function_drop,
                            font       = (default_font, round(12 * font_scale),"bold"), 
                            anchor     = "n",
                            relief     = "flat",
                            justify    = "left",
                            background = "#181818",
                            foreground = "#D3D3D3",
                            wraplength = 750 - 10)
    video_info_space.place(x = left_bar_width + drag_drop_width/2 - 750/2,
                               y = drag_drop_height/2 - 400/2,
                               width  = 750,
                               height = 380)

def show_list_images_in_GUI_with_drag_drop(image_list_prepared):
    clear_drag_drop_background()
    final_string = "\n"
    counter_img = 0
    for elem in image_list_prepared:
        counter_img += 1
        if counter_img <= 16:
            # add first 16 files in list
            img        = cv2.imread(elem.strip())
            width      = round(img.shape[1])
            height     = round(img.shape[0])
            img_name   = str(elem.split("/")[-1])
            
            final_string += ( img_name + " | [" + str(width) + "x" + str(height) + "]"+ "\n" )         
        else:
            final_string +=  "and others... \n"
            break

    list_height = 420
    list_width  = 750
    
    list_header = ttk.Label(root,
                            text       = " ❏  Image list ",
                            ondrop     = function_drop,
                            font       = (default_font, round(12 * font_scale), "bold"), #11
                            anchor     = "center",
                            relief     = "flat",
                            justify    = "center",
                            background = "#181818",
                            foreground = "#D3D3D3")
    list_header.place(x = left_bar_width + drag_drop_width/2 - list_width/2,
                            y = drag_drop_height/2 - list_height/2 - 45,
                            width  = 200,
                            height = 36)

    multiple_images_list = ttk.Label(root,
                            text       = final_string,
                            ondrop     = function_drop,
                            font       = (default_font, round(11 * font_scale)), #9
                            anchor     = "n",
                            relief     = "flat",
                            justify    = "left",
                            background = "#181818",
                            foreground = "#D3D3D3",
                            wraplength = list_width - 10)
    multiple_images_list.place(x = left_bar_width + drag_drop_width/2 - list_width/2,
                            y = drag_drop_height/2 - list_height/2,
                            width  = list_width,
                            height = list_height)

    # then image counter
    multiple_images_label = ttk.Label(root,
                            text       = str(len(image_list_prepared)) + ' images',
                            ondrop     = function_drop,
                            font       = (default_font, round(12 * font_scale), "bold"),
                            anchor     = "center",
                            relief     = "flat",
                            justify    = "center",
                            background = "#181818",
                            foreground = "#D3D3D3")
    multiple_images_label.place(x = left_bar_width + drag_drop_width/2 + 175,
                        y = drag_drop_height/2 + 220,
                        width  = 200,
                        height = 36)

def show_image_in_GUI_with_drag_drop(image_to_show):
    global image

    image_to_show = image_to_show.replace('{', '').replace('}','')

    # 1) resize image to fit the UI
    image_to_show_resized = resize_image_to_show(image_to_show)
    
    # 2) clean the background
    clear_drag_drop_background()

    # 3) show the resized image in the UI
    image  = tk.PhotoImage(file = image_to_show_resized)
    drag_drop_and_images = ttk.Label(root,
                            text    = "",
                            image   = image,
                            ondrop     = function_drop,
                            anchor     = "center",
                            relief     = "flat",
                            justify    = "center",
                            background = drag_drop_background,
                            foreground = "#202020")
    drag_drop_and_images.place(x      = left_bar_width + drag_drop_width/2 - show_image_width/2,
                            y      = drag_drop_height/2 - show_image_height/2 - image_text_height+1,
                            width  = show_image_width,
                            height = show_image_height)
    
    # 4) show the image file information in the UI
    place_fileName_label(image_to_show)
    
    # 5) delete the resized temp image 
    if "temp.png" in image_to_show_resized:
        os.remove("temp.png")
        
def place_fileName_label(image_path):
    path_as_list = image_path.split("/")
    img_name     = str(path_as_list[-1])
    img        = cv2.imread(image_path.replace("{", "").replace("}", ""))
    width      = round(img.shape[1])
    height     = round(img.shape[0])
    file_size  = Path(image_path).stat().st_size

    file_name_string.set(img_name 
                         + " | [" + str(width) + "x" + str(height) + "]" 
                         + " | "  + str(truncate(file_size / 1048576, 2)) + " MB")
    single_image_info = ttk.Label(root,
                            font = (default_font, round(10 * font_scale), "bold"),
                            textvar    = file_name_string,
                            relief     = "flat",
                            justify    = "center",
                            background = "#181818",
                            foreground = "#D3D3D3",
                            anchor     = "center")
                            
    single_image_info.place(x = left_bar_width + drag_drop_width/2 - image_text_width/2,
                    y = drag_drop_height - image_text_height - 24,
                    width  = image_text_width,
                    height = image_text_height + 5)

# ---------------------- Buttons ----------------------

def place_upscale_button():
    upscale_button_width = button_width + 50

    Upscale_button            = tk.Button(root)
    Upscale_button["bg"]      = "#01aaed"
    ft = tkFont.Font(family   = default_font, 
                     size     = round(12 * font_scale), 
                     weight   = 'bold')
    Upscale_button["font"]    = ft
    Upscale_button["fg"]      = "#202020"
    Upscale_button["justify"] = "center"
    Upscale_button["text"]    = "Upscale"
    Upscale_button["relief"]  = "flat"
    Upscale_button.place(x      = left_bar_width/2 - upscale_button_width/2,
                         y      = left_bar_height - 60,
                         width  = upscale_button_width,
                         height = 40)
    Upscale_button["command"] = lambda : upscale_button_command()

def place_stop_button():
    stop_button_width = button_width + 50

    Stop_button            = tk.Button(root)
    Stop_button["bg"]      = "#FF4433"
    ft = tkFont.Font(family   = default_font, 
                     size     = round(12 * font_scale), 
                     weight   = 'bold')
    Stop_button["font"]    = ft
    Stop_button["fg"]      = "#202020"
    Stop_button["justify"] = "center"
    Stop_button["text"]    = "Stop upscaling"
    Stop_button["relief"]  = "flat"
    Stop_button.place(x      = left_bar_width/2 - stop_button_width/2,
                         y      = left_bar_height - 60,
                         width  = stop_button_width,
                         height = 40)
    Stop_button["command"] = lambda : stop_button_command()
    
def place_BSRGAN_button(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight = "bold")
    BSRGAN_button            = tk.Button(root)
    BSRGAN_button["anchor"]  = "center"
    BSRGAN_button["bg"]      = background_color
    BSRGAN_button["font"]    = ft
    BSRGAN_button["fg"]      = text_color
    BSRGAN_button["justify"] = "center"
    BSRGAN_button["text"]    = "BSRGAN"
    BSRGAN_button["relief"]  = "flat"
    BSRGAN_button["activebackground"] = "#ffbf00"
    BSRGAN_button.place(x = left_bar_width/2 - button_width/2 ,
                    y = button_1_y,
                    width  = button_width,
                    height = button_height)
    BSRGAN_button["command"] = lambda input = "BSRGAN" : choose_model_BSRGAN(input)
    
def place_RealSR_JPEG_button(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight = "bold")
    RealSR_JPEG_button            = tk.Button(root)
    RealSR_JPEG_button["anchor"]  = "center"
    RealSR_JPEG_button["bg"]      = background_color
    RealSR_JPEG_button["font"]    = ft
    RealSR_JPEG_button["fg"]      = text_color
    RealSR_JPEG_button["justify"] = "center"
    RealSR_JPEG_button["text"]    = "RealSR_JPEG"
    RealSR_JPEG_button["relief"]  = "flat"
    RealSR_JPEG_button["activebackground"] = "#ffbf00"
    RealSR_JPEG_button.place(x = left_bar_width/2 - button_width/2,
                    y = button_2_y,
                    width  = button_width,
                    height = button_height)
    RealSR_JPEG_button["command"] = lambda input = "RealSR_JPEG" : choose_model_RealSR_JPEG(input)

def place_BSRGAN_and_RealSR_JPEG_button(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight  = "bold")
    Both_AI_button            = tk.Button(root)
    Both_AI_button["anchor"]  = "center"
    Both_AI_button["bg"]      = background_color
    Both_AI_button["font"]    = ft
    Both_AI_button["fg"]      = text_color
    Both_AI_button["justify"] = "center"
    Both_AI_button["text"]    = "wip :)"
    Both_AI_button["relief"]  = "flat"
    Both_AI_button["activebackground"] = "#ffbf00"
    Both_AI_button.place(x = left_bar_width/2 - button_width/2 ,
                    y = button_3_y,
                    width  = button_width,
                    height = button_height)
    # WIP
    # Both_AI_button["command"] = lambda input = "BSRGAN" : choose_model_BSRGAN(input)
 
def choose_model_BSRGAN(choosed_model):
    global AI_model
    AI_model = choosed_model
    
    default_button_color  = "#484848"
    default_text_color    = "#DCDCDC"
    selected_button_color = "#ffbf00"
    selected_text_color   = "#202020"
    
    place_BSRGAN_button(selected_button_color, selected_text_color) # changing
    place_RealSR_JPEG_button(default_button_color, default_text_color)

def choose_model_RealSR_JPEG(choosed_model):
    global AI_model
    AI_model = choosed_model
    
    default_button_color  = "#484848"
    default_text_color    = "#DCDCDC"
    selected_button_color = "#ffbf00"
    selected_text_color   = "#202020"
    
    place_RealSR_JPEG_button(selected_button_color, selected_text_color) # changing
    place_BSRGAN_button(default_button_color, default_text_color)

def place_upscale_factor_button_x2(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight  = "bold")
    Factor_x2_button            = tk.Button(root)
    Factor_x2_button["anchor"]  = "center"
    Factor_x2_button["bg"]      = background_color
    Factor_x2_button["font"]    = ft
    Factor_x2_button["fg"]      = text_color
    Factor_x2_button["justify"] = "center"
    Factor_x2_button["text"]    = "x2"
    Factor_x2_button["relief"]  = "flat"
    Factor_x2_button["activebackground"] = "#ffbf00"
    Factor_x2_button.place(x = left_bar_width/2 - button_width/2,
                           y = 320,
                           width  = button_width,
                           height = button_height)
    Factor_x2_button["command"] = lambda : choose_upscale_x2()

def place_upscale_factor_button_x3(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight  = "bold")
    Factor_x2_button            = tk.Button(root)
    Factor_x2_button["anchor"]  = "center"
    Factor_x2_button["bg"]      = background_color
    Factor_x2_button["font"]    = ft
    Factor_x2_button["fg"]      = text_color
    Factor_x2_button["justify"] = "center"
    Factor_x2_button["text"]    = "x3"
    Factor_x2_button["relief"]  = "flat"
    Factor_x2_button["activebackground"] = "#ffbf00"
    Factor_x2_button.place(x = left_bar_width/2 - button_width/2,
                           y = 370,
                           width  = button_width,
                           height = button_height)
    Factor_x2_button["command"] = lambda : choose_upscale_x3()

def place_upscale_factor_button_x4(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight  = "bold")
    Factor_x4_button            = tk.Button(root)
    Factor_x4_button["anchor"]  = "center"
    Factor_x4_button["bg"]      = background_color
    Factor_x4_button["font"]    = ft
    Factor_x4_button["fg"]      = text_color
    Factor_x4_button["justify"] = "center"
    Factor_x4_button["text"]    = "x4"
    Factor_x4_button["relief"]  = "flat"
    Factor_x4_button["activebackground"] = "#ffbf00"
    Factor_x4_button.place(x = left_bar_width/2 - button_width/2,
                           y = 420,
                           width  = button_width,
                           height = button_height)
    Factor_x4_button["command"] = lambda : choose_upscale_x4()

def choose_upscale_x2():
    global upscale_factor
    upscale_factor = 2

    default_button_color  = "#484848"
    default_text_color    = "#DCDCDC"
    selected_button_color = "#ffbf00"
    selected_text_color   = "#202020"

    place_upscale_factor_button_x2(selected_button_color, selected_text_color)   # selected
    place_upscale_factor_button_x3(default_button_color, default_text_color)     # not selected
    place_upscale_factor_button_x4(default_button_color, default_text_color)     # not selected

def choose_upscale_x3():
    global upscale_factor
    upscale_factor = 3

    default_button_color  = "#484848"
    default_text_color    = "#DCDCDC"
    selected_button_color = "#ffbf00"
    selected_text_color   = "#202020"

    place_upscale_factor_button_x2(default_button_color, default_text_color)     # not selected
    place_upscale_factor_button_x3(selected_button_color, selected_text_color)   # selected
    place_upscale_factor_button_x4(default_button_color, default_text_color)     # not selected

def choose_upscale_x4():
    global upscale_factor
    upscale_factor = 4

    default_button_color  = "#484848"
    default_text_color    = "#DCDCDC"
    selected_button_color = "#ffbf00"
    selected_text_color   = "#202020"

    place_upscale_factor_button_x2(default_button_color, default_text_color)     # not selected
    place_upscale_factor_button_x3(default_button_color, default_text_color)     # not selected
    place_upscale_factor_button_x4(selected_button_color, selected_text_color)   # selected

def place_upscale_backend_cuda(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight  = "bold")
    Backend_cpu_button            = tk.Button(root)
    Backend_cpu_button["anchor"]  = "center"
    Backend_cpu_button["justify"] = "center"
    Backend_cpu_button["bg"]      = background_color
    Backend_cpu_button["font"]    = ft
    Backend_cpu_button["fg"]      = text_color
    Backend_cpu_button["text"]    = "gpu"
    Backend_cpu_button["relief"]  = "flat"
    Backend_cpu_button["activebackground"] = "#ffbf00"
    Backend_cpu_button.place(x = left_bar_width/2 - button_width/2,
                           y = 520,
                           width  = button_width,
                           height = button_height)
    Backend_cpu_button["command"] = lambda : choose_backend_cuda()

def place_upscale_backend_cpu(background_color, text_color):
    ft = tkFont.Font(family  = default_font, 
                     size    = round(11 * font_scale),
                     weight  = "bold")
    Backend_cpu_button            = tk.Button(root)
    Backend_cpu_button["anchor"]  = "center"
    Backend_cpu_button["justify"] = "center"
    Backend_cpu_button["bg"]      = background_color
    Backend_cpu_button["font"]    = ft
    Backend_cpu_button["fg"]      = text_color
    Backend_cpu_button["text"]    = "cpu"
    Backend_cpu_button["relief"]  = "flat"
    Backend_cpu_button["activebackground"] = "#ffbf00"
    Backend_cpu_button.place(x = left_bar_width/2 - button_width/2,
                           y = 570,
                           width  = button_width,
                           height = button_height)
    Backend_cpu_button["command"] = lambda : choose_backend_cpu()

def choose_backend_cpu():
    global device
    device = "cpu"

    default_button_color  = "#484848"
    default_text_color    = "#DCDCDC"
    selected_button_color = "#ffbf00"
    selected_text_color   = "#202020"

    place_upscale_backend_cpu(selected_button_color, selected_text_color)  
    place_upscale_backend_cuda(default_button_color, default_text_color) 

def choose_backend_cuda():
    if torch.cuda.is_available():
        global device
        device = "cuda"

        default_button_color  = "#484848"
        default_text_color    = "#DCDCDC"
        selected_button_color = "#ffbf00"
        selected_text_color   = "#202020"
    
        place_upscale_backend_cpu(default_button_color, default_text_color) 
        place_upscale_backend_cuda(selected_button_color, selected_text_color)
    else: 
        error_root    = tkinterDnD.Tk()
        ErrorMessage(error_root, "no_cuda_found")

# ---------------------- /Buttons ----------------------

# ---------------------- /GUI related ---------------------- 

# ---------------------- /Functions ----------------------

def apply_windows_dark_bar(window_root):
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd                 = ctypes.windll.user32.GetParent(window_root.winfo_id())
        value                = ctypes.c_int(2)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value))

class ErrorMessage():
    def __init__(self, error_root, error_type):
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
        scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

        if scaleFactor == 1.0:
            font_scale = 1.25
        elif scaleFactor == 1.25:
            font_scale = 1.0
        else:
            font_scale = 0.85

        error_root.title(" Error message ")
        width        = 500
        height       = 525
        screenwidth  = error_root.winfo_screenwidth()
        screenheight = error_root.winfo_screenheight()
        alignstr     = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        error_root.geometry(alignstr)
        error_root.resizable(width=False, height=False)

        error_root.iconbitmap(create_void_logo())

        window_width  = 500
        window_height = 525

        if error_type == "upscale_problem":
            error_string = "Memory\nerror"
            error_suggest = ( " Some error occured while upscaling, try to: \n\n"
                             + "  set Upscale Factor to x2 or x3  \n" 
                             + "  set AI Backend to <cpu>  " )
        elif error_type == "no_cuda_found":
            error_string = "Cuda not found\nwarning"
            error_suggest = (" Cuda compatible GPU not found, try to: \n\n"
                             + "  set AI Backend to <cpu>  ")

        ft = tkFont.Font(family = default_font, 
                         size   =  int(12 * font_scale),
                         weight = "bold")

        Error_container            = tk.Label(error_root)
        Error_container["anchor"]  = "center"
        Error_container["justify"] = "center"
        Error_container["font"]    = ft
        Error_container["bg"]      = "#FF4433"
        Error_container["fg"]      = "#202020"
        Error_container["text"]    = error_string
        Error_container["relief"]  = "flat"
        Error_container.place(x = 0,
                              y = 0,
                              width = window_width, 
                              height = window_height/3)            

        Suggest_container            = tk.Label(error_root)
        Suggest_container["anchor"]  = "center"
        Suggest_container["justify"] = "center"
        Suggest_container["font"]    = ft
        Suggest_container["bg"]      = "#202020"
        Suggest_container["fg"]      = "#FF4433" 
        Suggest_container["text"]    = error_suggest
        Suggest_container["relief"]  = "flat"
        Suggest_container.place(x = 0,
                                y = window_height/3,
                                width  = window_width, 
                                height = (window_height/3)*2)

        error_root.attributes('-topmost',True)
        apply_windows_dark_bar(error_root)
        
        error_root.update()
        error_root.mainloop()

class App:
    def __init__(self, root):
        root.title(" QualityScaler " + version)
        width        = window_width
        height       = window_height
        screenwidth  = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr     = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        logo = PhotoImage(file=find_file_by_relative_path("logo.png"))
        root.iconphoto(False, logo)

        # BIG BLACK BAR
        Left_container            = tk.Label(root)
        Left_container["anchor"]  = "e"
        Left_container["bg"]      = "#202020"
        Left_container["relief"]  = "flat"
        Left_container.place(x=0,y=0,width = left_bar_width, height = left_bar_height)
        
        # TITLE
        ft = tkFont.Font(family=default_font, 
                         size = round(16 * font_scale),
                         weight = "bold"),
        Title            = tk.Label(root)
        Title["bg"]      = "#202020"
        Title["font"]    = ft
        Title["fg"]      = "#DA70D6"
        Title["anchor"]  = "center" 
        Title["text"]    = "QualityScaler"
        Title.place(x = 15,
                    y = 3,
                    width  = left_bar_width/2,
                    height = 55)
        
        global logo_git
        logo_git = PhotoImage(file=find_file_by_relative_path("github_logo.png"))
        logo_git_label            = tk.Button(root)
        logo_git_label['image']   = logo_git
        logo_git_label["justify"] = "center"
        logo_git_label["bg"]      = "#585858"
        logo_git_label["relief"]  = "flat"
        logo_git_label["activebackground"] = "grey"
        logo_git_label.place(x    = left_bar_width - 165,
                             y      = 13,
                             width  = 37,
                             height = 37)
        logo_git_label["command"] = lambda : opengithub()

        global logo_paypal
        logo_paypal = PhotoImage(file=find_file_by_relative_path("paypal_logo.png"))
        logo_paypal_label                     = tk.Button(root)
        logo_paypal_label['image']            = logo_paypal
        logo_paypal_label["justify"]          = "center"
        logo_paypal_label["bg"]               = "black"
        logo_paypal_label["relief"]           = "flat"
        logo_paypal_label["activebackground"] = "grey"
        logo_paypal_label["borderwidth"]      = 1
        logo_paypal_label.place(x      = left_bar_width - 115,
                                y      = 13,
                                width  = 37,
                                height = 37)
        logo_paypal_label["command"] = lambda : openpaypal()

        global logo_patreon
        logo_patreon = PhotoImage(file=find_file_by_relative_path("patreon_logo.png"))
        logo_patreon_label                     = tk.Button(root)
        logo_patreon_label['image']            = logo_patreon
        logo_patreon_label["justify"]          = "center"
        logo_patreon_label["bg"]               = "black"
        logo_patreon_label["relief"]           = "flat"
        logo_patreon_label["activebackground"] = "grey"
        logo_patreon_label["borderwidth"]      = 1
        logo_patreon_label.place(x      = left_bar_width - 65,
                                y      = 13,
                                width  = 37,
                                height = 37)
        logo_patreon_label["command"] = lambda : openpatreon()

        ft = tkFont.Font(family = default_font,
                         size   = round(12 * font_scale), 
                         weight = "bold")      
        IA_selection_title            = tk.Label(root)
        IA_selection_title["bg"]      = "#202020"
        IA_selection_title["font"]    = ft
        IA_selection_title["fg"]      = "#DCDCDC" 	
        IA_selection_title["anchor"]  = "w" 
        IA_selection_title["justify"] = "center"
        IA_selection_title["relief"]  = "flat"
        IA_selection_title["text"]    = " ◪  AI model "
        IA_selection_title.place(x      = left_bar_width/2 - 174,
                                 y      = 70,
                                 width  = 348,
                                 height = 40)

        # BUTTONS
        default_button_color  = "#484848"
        default_text_color    = "#DCDCDC"
        selected_button_color = "#ffbf00"
        selected_text_color   = "#202020"
        
        place_RealSR_JPEG_button(default_button_color, default_text_color)  #default selected                  
        place_BSRGAN_button(selected_button_color, selected_text_color)       
        place_BSRGAN_and_RealSR_JPEG_button(default_button_color, default_text_color) 

        Upscale_fact_selection_title            = tk.Label(root)
        Upscale_fact_selection_title["bg"]      = "#202020"
        Upscale_fact_selection_title["font"]    = ft
        Upscale_fact_selection_title["fg"]      = "#DCDCDC" 	
        Upscale_fact_selection_title["anchor"]  = "w" 
        Upscale_fact_selection_title["justify"] = "center"
        Upscale_fact_selection_title["relief"]  = "flat"
        Upscale_fact_selection_title["text"]    = " ⤮  Upscale factor "
        Upscale_fact_selection_title.place(x      = left_bar_width/2 - 175,
                                           y      = 270,
                                           width  = 155,
                                           height = 40)

        place_upscale_factor_button_x2(selected_button_color, selected_text_color)
        place_upscale_factor_button_x3(default_button_color, default_text_color)
        place_upscale_factor_button_x4(default_button_color, default_text_color)
      

        Upscale_backend_selection_title            = tk.Label(root)
        Upscale_backend_selection_title["bg"]      = "#202020"
        Upscale_backend_selection_title["font"]    = ft
        Upscale_backend_selection_title["fg"]      = "#DCDCDC" 	
        Upscale_backend_selection_title["anchor"]  = "w" 
        Upscale_backend_selection_title["justify"] = "center"
        Upscale_backend_selection_title["relief"]  = "flat"
        Upscale_backend_selection_title["text"]    = " ⍚  AI backend "
        Upscale_backend_selection_title.place(x      = left_bar_width/2 - 175,
                                           y      = 470,
                                           width  = 145,
                                           height = 40)

        global device
        if torch.cuda.is_available():
            place_upscale_backend_cpu(default_button_color, default_text_color)
            place_upscale_backend_cuda(selected_button_color, selected_text_color)
            device = "cuda"
        else:
            device = "cpu"
            place_upscale_backend_cpu(selected_button_color, selected_text_color)
            place_upscale_backend_cuda(default_button_color, default_text_color)
     
        
        # MESSAGE
        error_message_label = ttk.Label(root,
                              font       = (default_font, round(11 * font_scale), "bold"),
                              textvar    = info_string,
                              relief     = "flat",
                              justify    = "center",
                              background = "#202020",
                              foreground = "#ffbf00",
                              anchor     = "center")
        error_message_label.place(x      = 0,
                                  y      = 625,
                                  width  = left_bar_width,
                                  height = 30)
        
        # UPSCALE BUTTON
        place_upscale_button()

        # DRAG & DROP WIDGET
        ft = tkFont.Font(family  = default_font, 
                     size    = round(13 * font_scale),
                     weight = "bold")

        drag_drop = ttk.Label(root,
                              text    = " DROP FILE HERE \n\n" 
                                      + " ⥥ \n\n"
                                      + " IMAGES  - jpg png tif bmp webp -                  \n\n" 
                                      + " VIDEOS  - mp4 webm mkv flv gif avi mov -  \n\n",
                              ondrop     = function_drop,
                              font       = ft,
                              anchor     = "center",
                              relief     = "flat",
                              justify    = "center",
                              background = drag_drop_background,
                              foreground = drag_drop_text_color)
        drag_drop.place(x=left_bar_width,y=0,width = drag_drop_width, height = drag_drop_height)
        apply_windows_dark_bar(root)

if __name__ == "__main__":
    multiprocessing.freeze_support()
     
    root              = tkinterDnD.Tk()
    file_name_string  = tk.StringVar()
    info_string       = tk.StringVar()

    app = App(root)
    root.update()
    root.mainloop()
