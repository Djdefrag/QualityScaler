
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
from timeit import default_timer as timer
from tkinter import *
from tkinter import ttk

import cv2
import moviepy.video.io.ImageSequenceClip
import numpy as np
import sv_ttk
import tkinterDnD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from image_slicer import join
from image_slicer import slice as img_cutter
from PIL import Image
from win32mica import MICAMODE, ApplyMica

ctypes.windll.shcore.SetProcessDpiAwareness(True)
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

if scaleFactor == 1.0:
    font_scale = 1.2
elif scaleFactor == 1.25:
    font_scale = 1.0
else:
    font_scale = 0.8

version   = "2.0.0"
branch    = "stable"

paypalme  = "https://www.paypal.com/paypalme/jjstd/5"
githubme  = "https://github.com/Djdefrag/QualityScaler"
patreonme = "https://www.patreon.com/Djdefrag"

image_path        = "no file"
AI_model          = "BSRGAN"
device            = "dml"
input_video_path  = ""
upscale_factor    = 2
single_file       = False
multiple_files    = False
video_files       = False
multi_img_list    = []
video_frames_list = []
video_frames_upscaled_list = []

tiles_resolution = 500

supported_file_list = ['.jpg', '.jpeg', '.JPG', '.JPEG',
                       '.png', '.PNG',
                       '.webp', '.WEBP',
                       '.bmp', '.BMP',
                       '.tif', '.tiff', '.TIF', '.TIFF',
                       '.mp4', '.MP4',
                       '.webm', '.WEBM',
                       '.mkv', '.MKV',
                       '.flv', '.FLV',
                       '.gif', '.GIF',
                       '.m4v', ',M4V',
                       '.avi', '.AVI',
                       '.mov', '.MOV',
                       '.qt',
                       '.3gp', '.mpg', '.mpeg']

supported_video_list = ['.mp4', '.MP4',
                        '.webm', '.WEBM',
                        '.mkv', '.MKV',
                        '.flv', '.FLV',
                        '.gif', '.GIF',
                        '.m4v', ',M4V',
                        '.avi', '.AVI',
                        '.mov', '.MOV',
                        '.qt',
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
                           '.sys', '.tmp', '.xlt', '.avif']


# ---------------------- Dimensions ----------------------

default_font         = 'Calibri'
background_color     = "#202020"
window_width         = 1300
window_height        = 725
left_bar_width       = 410
left_bar_height      = window_height
drag_drop_width      = window_width - left_bar_width
drag_drop_height     = window_height
button_width         = 270
button_height        = 34
show_image_width     = drag_drop_width * 0.8
show_image_height    = drag_drop_width * 0.6
image_text_width     = drag_drop_width * 0.8
image_text_height    = 34
button_1_y           = 150
button_2_y           = 260
button_3_y           = 370
button_4_y           = 480
drag_drop_text_color = "#E0E0E0"

not_selected_button_color  = "#484848"
not_selected_text_color    = '#F5F5F5'
selected_button_color      = "#ffbf00"
selected_text_color        = "#202020"

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
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

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
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(
                fea, scale_factor=2, mode='nearest')))
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

def image_read_uint(path, n_channels=3):
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

def image_save(img, img_path):
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
    _, ICON_PATH = tempfile.mkstemp()
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

def delete_tiles(tiles):
    # remove tiles file
    for tile in tiles:
        os.remove(tile.filename)

def resize_image_to_show(image_to_prepare):

    old_image     = cv2.imread(image_to_prepare)
    actual_width  = old_image.shape[1]
    actual_height = old_image.shape[0]

    if actual_width >= actual_height:
        max_val = actual_width
        max_photo_resolution = show_image_width
    else:
        max_val = actual_height
        max_photo_resolution = show_image_height

    if max_val >= max_photo_resolution:
        downscale_factor = max_val/max_photo_resolution
        new_width        = round(old_image.shape[1]/downscale_factor)
        new_height       = round(old_image.shape[0]/downscale_factor)
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"
    else:
        new_width        = round(old_image.shape[1])
        new_height       = round(old_image.shape[0])
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_LINEAR)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"

def prepare_output_filename(img, AI_model, upscale_factor):
    result_path = (img.replace("_resized.png", "").replace(".png", "") +
                   "_" + AI_model + "_x" + str(upscale_factor) + ".png")
    return result_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def adapt_image_for_deeplearning(img, device):
    if 'cpu' in device:
        backend = torch.device('cpu')
    elif 'dml' in device:
        backend = torch.device('dml')

    img = image_read_uint(img, n_channels=3)
    img = uint2tensor4(img)
    img = img.to(backend, non_blocking=True)
    
    return img

def resize_image(image_path, upscale_factor):
    new_image_path = image_path.replace(".png", "_resized.png")
    
    old_image = cv2.imread(image_path)

    if upscale_factor == 1:
        # no upscale
        # resize image dividing by 4
        new_width   = round(old_image.shape[1]/4)
        new_height  = round(old_image.shape[0]/4)
        resized_image = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_LINEAR)
        
        cv2.imwrite(new_image_path, resized_image)
        return new_image_path

    if upscale_factor == 2:
        # upscale x2
        # resize image dividing by 2
        new_width   = round(old_image.shape[1]/2)
        new_height  = round(old_image.shape[0]/2)
        resized_image = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_LINEAR)
        
        cv2.imwrite(new_image_path, resized_image)
        return new_image_path

    elif upscale_factor == 3:
        # upscale x3
        # resize image subtracting 1/4 in resolution
        # ex. 1000 --> 750 --> 3000
        width_to_remove  = round(old_image.shape[1]/4)
        height_to_remove = round(old_image.shape[0]/4)

        new_width     = round(old_image.shape[1] - width_to_remove)
        new_height    = round(old_image.shape[0] - height_to_remove)
        resized_image = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_LINEAR)
        
        cv2.imwrite(new_image_path, resized_image)
        return new_image_path

    elif upscale_factor == 4:
        # upscale by 4
        cv2.imwrite(new_image_path, old_image)
        return new_image_path

def resize_multiple_images(image_list, upscale_factor):
    files_to_delete   = []
    downscaled_images = []

    for image in image_list:
        img_downscaled = resize_image(image, upscale_factor)
        downscaled_images.append(img_downscaled)
        # if the image has been resized, 
        # add it to the list of files 
        # to delete after upscale
        if "_resized" in img_downscaled:
            files_to_delete.append(img_downscaled)

    return downscaled_images, files_to_delete

def update_log_file(text_to_insert):
    log_file_name   = "QualityScaler.log"
    with open(log_file_name,'w') as log_file:
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name   = "QualityScaler.log"
    with open(log_file_name,'r') as log_file:
        step = log_file.readline()
    log_file.close()
    return step

# ----------------------- /Utils ------------------------

def upscale_image_and_save(img, model, result_path, device, tiles_resolution):
    # 1) calculating slice number
    img_tmp          = cv2.imread(img)
    image_resolution = max(img_tmp.shape[1], img_tmp.shape[0])
    num_tiles        = image_resolution/tiles_resolution

    if num_tiles <= 1:
        # if the image fits entirely
        # in VRAM -> no tiles needed
        with torch.no_grad():
            # 2) upscale image without tiling
            img_adapted  = adapt_image_for_deeplearning(img, device)
            img_upscaled = tensor2uint(model(img_adapted))
        
        # 3) save upscaled image
        image_save(img_upscaled, result_path)
    else:
        # tiles needed
        num_tiles = round(num_tiles)
        if (num_tiles % 2) != 0:
            num_tiles += 1
        num_tiles = round(num_tiles * 2)

        # 2) divide the image in tiles
        tiles = slice_image(img, num_tiles)

        # 3) upscale each tiles
        with torch.no_grad():
            for tile in tiles:
                tile_adapted  = adapt_image_for_deeplearning(tile.filename, device)
                tile_upscaled = tensor2uint(model(tile_adapted))
                image_save(tile_upscaled, tile.filename)
                tile.image = Image.open(tile.filename)
                tile.coords = (tile.coords[0]*4, tile.coords[1]*4)

        # 4) then reconstruct the image by tiles upscaled
        image_upscaled = reunion_image(tiles)

        # 5) remove tiles file
        delete_tiles(tiles)

        # 6) save reconstructed image
        cv2.imwrite(result_path, image_upscaled)

def optimize_torch(device):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    if 'cpu' in device:
        cpu_numb = os.cpu_count()
        torch.set_num_threads(round(cpu_numb*0.8))

def prepare_torch_model(AI_model, device):
    if 'cpu' in device:
        backend = torch.device('cpu')
    elif 'dml' in device:
        backend = torch.device('dml')

    model_path = find_file_by_relative_path(AI_model + ".pth")

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for _, v in model.named_parameters():
        v.requires_grad = False
    
    model = model.to(backend, non_blocking=True)

    return model

def process_upscale_multiple_images_torch(image_list, AI_model, upscale_factor, device, tiles_resolution):
    try:
        update_log_file('Preparing for upscaling')
        
        start = timer()
        optimize_torch(device)

        # 0) convert images to png
        image_list = convert_multi_images_to_png(image_list)
        
        # 1) resize images, temp file to delete
        downscaled_images, files_to_delete = resize_multiple_images(image_list, upscale_factor)
        
        how_many_images = len(downscaled_images)
        done_images     = 0

        # 2) update the log file
        update_log_file('Upscaling')

        if 'Fused' in AI_model:
            # double images to upscale 
            # because images must be 
            # upscaled with both AI
            how_many_images = how_many_images * 2 

            upscaled_BSRGAN_list  = []
            upscaled_REAL_SR_list = []

            # < Bsrgan phase >

            # 3) define the model BSRGAN 
            model = prepare_torch_model('BSRGAN', device)

            for img in downscaled_images:
                result_path = prepare_output_filename(img, 'BSRGAN', upscale_factor)
                # 4) upscale the image with BSRGAN
                upscale_image_and_save(img, model, result_path, device, tiles_resolution)
                # 5) add image to BSRGAN list
                upscaled_BSRGAN_list.append(result_path)
                # 6) update counter and write in log file
                done_images += 1
                update_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
            
            # < Real_sr phase >

            # 7) define the model REAL_SR
            model = prepare_torch_model('RealSR_JPEG', device)
            for img in downscaled_images:
                result_path = prepare_output_filename(img, 'RealSR_JPEG', upscale_factor)
                # 8) upscale the image with Real_SR-jpeg
                upscale_image_and_save(img, model, result_path, device, tiles_resolution)
                # 9) add image to Real_SR list
                upscaled_REAL_SR_list.append(result_path)
                # 10) update counter and write in log file
                done_images += 1
                update_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
            
            # < Fusion phase >
            update_log_file("Merging images")

            # 11) now fuse the images of both models
            BSRGAN_weight  = 0.60
            Real_SR_weight = 0.40
            index = 0

            for img in downscaled_images:
                # 12) get images by index
                BSRGAN_image  = cv2.imread(upscaled_BSRGAN_list[index])
                Real_SR_image = cv2.imread(upscaled_REAL_SR_list[index])

                result_path = prepare_output_filename(img, AI_model, upscale_factor)

                # 13) fuse images and save
                fused_image = cv2.addWeighted(BSRGAN_image,
                                              BSRGAN_weight,
                                              Real_SR_image,
                                              Real_SR_weight,
                                              0.0)
                cv2.imwrite(result_path, fused_image)
                index += 1

            # 14) update log file, upscaling finished
            update_log_file("Upscale completed [" + str(round(timer() - start)) + " sec.]")
        
        else:
            # >>> Single model steps <<<

            # 3) define the model
            model = prepare_torch_model(AI_model, device)

            for img in downscaled_images:
                result_path = prepare_output_filename(img, AI_model, upscale_factor)

                # 4) upscale the image
                upscale_image_and_save(img, model, result_path, device, tiles_resolution)

                # 5) update counter and log file
                done_images += 1
                update_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
                    
            # 6) update log file, upscaling finished
            update_log_file("Upscale completed [" + str(round(timer() - start)) + " sec.]")

        delete_list_of_files(files_to_delete)
    except:
        update_log_file('Error while upscaling')
        error_root = tkinterDnD.Tk()
        ErrorMessage(error_root, "upscale_problem")

def process_upscale_video_frames_torch(input_video_path, AI_model, upscale_factor, device, tiles_resolution):
    try:
        start = timer()

        # 0) extract frames from input video
        update_log_file('Extracting video frames')
        image_list = extract_frames_from_video(input_video_path)
        
        # 1) resize all images
        update_log_file('Preparing for upscaling')
        
        downscaled_images, _ = resize_multiple_images(image_list, upscale_factor)

        # 2) prepare variables
        update_log_file('Upscaling')
        how_many_images = len(downscaled_images)
        done_images     = 0
        
        optimize_torch(device)

        if 'Fused' in AI_model:

            # double because the image must be 
            # upscaled with both AI
            how_many_images = how_many_images * 2 
            
            video_frames_upscaled_list = []
            upscaled_BSRGAN_list  = []
            upscaled_REAL_SR_list = []

            # < Bsrgan phase >

            # 3) define the model BSRGAN 
            model = prepare_torch_model('BSRGAN', device)

            for img in downscaled_images:
                result_path = prepare_output_filename(img, 'BSRGAN', upscale_factor)

                # 4) upscale the image with BSRGAN
                upscale_image_and_save(img, model, result_path, device, tiles_resolution)

                # 5) add image to BSRGAN list
                upscaled_BSRGAN_list.append(result_path)

                # 6) update counter and write in log file
                done_images += 1
                update_log_file("Upscaled frames " + str(done_images) + "/" + str(how_many_images))
            
            # < Real_sr phase >

            # 7) define the model REAL_SR
            model = prepare_torch_model('RealSR_JPEG', device)

            for img in downscaled_images:
                result_path = prepare_output_filename(img, 'RealSR_JPEG', upscale_factor)

                # 8) upscale the image with Real_SR-jpeg
                upscale_image_and_save(img, model, result_path, device, tiles_resolution)

                # 9) add image to Real_SR list
                upscaled_REAL_SR_list.append(result_path)

                # 10) update counter and write in log file
                done_images += 1
                update_log_file("Upscaled frames " + str(done_images) + "/" + str(how_many_images))
            
            # < Fusion phase >
            
            update_log_file("Merging frames")
            
            # 11) now fuse the images of both models
            BSRGAN_weight  = 0.60
            Real_SR_weight = 0.40
            index = 0

            for img in downscaled_images:
                BSRGAN_image  = cv2.imread(upscaled_BSRGAN_list[index])
                Real_SR_image = cv2.imread(upscaled_REAL_SR_list[index])

                result_path = prepare_output_filename(img, AI_model, upscale_factor)
                video_frames_upscaled_list.append(result_path)

                # 12) fuse images and save
                fused_image = cv2.addWeighted(BSRGAN_image,
                                              BSRGAN_weight,
                                              Real_SR_image,
                                              Real_SR_weight,
                                              0.0)
                cv2.imwrite(result_path, fused_image)
                index += 1

            update_log_file("Processing upscaled video")
            
            # 13) reconstruct the video with upscaled frames
            video_reconstruction_by_frames(input_video_path, video_frames_upscaled_list, AI_model, upscale_factor)

            update_log_file("Upscale video completed [" + str(round(timer() - start)) + " sec.]")
        
        else:
            # >>> Single model steps <<<

            video_frames_upscaled_list = []

            # 1) define the model
            model = prepare_torch_model(AI_model, device)

            for img in downscaled_images:
                result_path = prepare_output_filename(img, AI_model, upscale_factor)
                video_frames_upscaled_list.append(result_path)

                # 2) upscale the image
                upscale_image_and_save(img, model, result_path, device, tiles_resolution)
                
                done_images += 1
                update_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))

            update_log_file("Processing upscaled video")
            
            # 3) reconstruct the video with upscaled frames
            video_reconstruction_by_frames(input_video_path, video_frames_upscaled_list, AI_model, upscale_factor)

            # 4) update log file, upscaling finished
            update_log_file("Upscale video completed [" + str(round(timer() - start)) + " sec.]")
    
    except:
        update_log_file('Error while upscaling')
        error_root = tkinterDnD.Tk()
        ErrorMessage(error_root, "upscale_problem")

def function_drop(event):
    global image_path
    global multiple_files
    global multi_img_list
    global video_files
    global single_file
    global input_video_path

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
                    input_video_path = str(event.data).replace("{", "").replace("}", "")
                    
                    show_video_info_with_drag_drop(input_video_path)

                    # reset variable
                    image_path = "no file"
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

                show_list_images_in_GUI(image_list_dropped)
                
                multi_img_list = image_list_dropped

                place_clean_button()

                # reset variable
                image_path = "no file"
                video_frames_list = []

            elif single_file:
                image_list_dropped = from_string_to_image_list(event)

                # convert images to png
                show_single_image_inGUI = threading.Thread(target = show_image_in_GUI,
                                                         args=(str(image_list_dropped[0]), 1),
                                                         daemon=True)
                show_single_image_inGUI.start()

                multi_img_list = image_list_dropped

                # reset variable
                image_path = "no file"
                video_frames_list = []

def extract_frames_from_video(video_path):
    create_temp_dir("QualityScaler_temp")
    num_frame = 0
    video_frames_list = []
    
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        num_frame += 1
        result_path = "QualityScaler_temp" + \
            os.sep + "frame_" + str(num_frame) + ".png"
        cv2.imwrite(result_path, frame)
        video_frames_list.append(result_path)
    cap.release()

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, video_frames_upscaled_list, AI_model, upscale_factor):
    # 1) get original video informations
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    path_as_list = input_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = input_video_path.replace(video_name, "")
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
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
        video_frames_upscaled_list, fps=frame_rate)
    clip.write_videofile(upscaled_video_name)

def check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number):
    all_supported  = True
    single_file    = False
    multiple_files = False
    video_files    = False
    more_than_one_video = False

    if not_supported_file_dropped_number > 0:
        all_supported = False

    if supported_file_dropped_number + not_supported_file_dropped_number == 1:
        single_file = True
    elif supported_file_dropped_number + not_supported_file_dropped_number > 1:
        multiple_files = True

    if supported_video_dropped_number == 1:
        video_files = True
        more_than_one_video = False
    elif supported_video_dropped_number > 1:
        video_files = True
        more_than_one_video = True

    return all_supported, single_file, multiple_files, video_files, more_than_one_video

def count_files_dropped(event):
    supported_file_dropped_number = 0
    not_supported_file_dropped_number = 0
    supported_video_dropped_number = 0

    # count compatible images files
    for file_type in supported_file_list:
        supported_file_dropped_number = supported_file_dropped_number + \
            str(event.data).count(file_type)

    # count compatible video files
    for file_type in supported_video_list:
        supported_video_dropped_number = supported_video_dropped_number + \
            str(event.data).count(file_type)

    # count not supported files
    for file_type in not_supported_file_list:
        not_supported_file_dropped_number = not_supported_file_dropped_number + \
            str(event.data).count(file_type)

    return supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number

def thread_check_steps_for_images( not_used_var, not_used_var2 ):
    time.sleep(2)
    try:
        while True:
            step = read_log_file()
            if "Upscale completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(1)
    except:
        place_upscale_button()

def thread_check_steps_for_videos( not_used_var, not_used_var2 ):
    time.sleep(2)
    try:
        while True:
            step = read_log_file()
            if "Upscale video completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(1)
    except:
        place_upscale_button()

def from_string_to_image_list(event):
    image_list = str(event.data).replace("{", "").replace("}", "")

    for file_type in supported_file_list:
        image_list = image_list.replace(file_type, file_type+"\n")

    image_list = image_list.split("\n")
    image_list.pop()  # to remove last void element

    return image_list

def convert_multi_images_to_png(image_list):
    converted_images = []
    for image in image_list:
        image = image.strip()
        converted_img = convert_image_png_and_save(image)
        converted_images.append(converted_img)

    return converted_images

def convert_image_png_and_save(image_to_prepare):
    image_to_prepare = image_to_prepare.replace("{", "").replace("}", "")
    if ".png" in image_to_prepare:
        return image_to_prepare
    else:
        new_image_path = image_to_prepare
        for file_type in supported_file_list:
            new_image_path = new_image_path.replace(file_type, ".png")

        image_to_convert = cv2.imread(image_to_prepare)
        cv2.imwrite(new_image_path, image_to_convert)
        return new_image_path

# ---------------------- GUI related ----------------------

def upscale_button_command():
    global image_path
    global multiple_files
    global process_upscale
    global thread_wait
    global upscale_factor
    global video_frames_list
    global video_files
    global video_frames_upscaled_list
    global input_video_path
    global device

    info_string.set("...")

    if video_files:
        place_stop_button()

        process_upscale = multiprocessing.Process(target = process_upscale_video_frames_torch,
                                                  args   = (input_video_path, 
                                                            AI_model, 
                                                            upscale_factor, 
                                                            device,
                                                            tiles_resolution))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_videos,
                                       args   = (upscale_factor, device), 
                                       daemon = True)
        thread_wait.start()

    elif multiple_files:
        place_stop_button()
        
        process_upscale = multiprocessing.Process(target = process_upscale_multiple_images_torch,
                                                    args   = (multi_img_list, 
                                                             AI_model, 
                                                             upscale_factor, 
                                                             device,
                                                             tiles_resolution))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                        args   = (upscale_factor, device), daemon = True)
        thread_wait.start()

    elif single_file:
        place_stop_button()

        process_upscale = multiprocessing.Process(target = process_upscale_multiple_images_torch,
                                                    args   = (multi_img_list, 
                                                             AI_model, 
                                                             upscale_factor, 
                                                             device,
                                                             tiles_resolution))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                       args   = (upscale_factor, device), daemon = True)
        thread_wait.start()

    elif "no file" in image_path:
        info_string.set("No file selected")
  
def stop_button_command():
    global process_upscale
    process_upscale.terminate()
    
    # this will stop thread that check upscaling steps
    update_log_file("Stopped upscaling") 

def clear_input_variables():
    global image_path
    global multi_img_list
    global video_frames_list
    global single_file
    global multiple_files
    global video_files

    # reset variable
    image_path        = "no file"
    multi_img_list    = []
    video_frames_list = []
    single_file       = False
    multiple_files    = False
    video_files       = False
    multi_img_list    = []
    video_frames_list = []

def clear_app_background():
    drag_drop = ttk.Label(root,
                          ondrop = function_drop,
                          relief = "flat",
                          background = background_color,
                          foreground = drag_drop_text_color)
    drag_drop.place(x = left_bar_width + 40, y=0,
                    width = drag_drop_width, height = drag_drop_height)

def place_drag_drop_widget():
    clear_input_variables()

    clear_app_background()

    ft = tkFont.Font(family = default_font,
                        size   = round(12 * font_scale),
                        weight = "bold")

    text_drop = (" DROP FILES HERE \n\n"
                + " ⥥ \n\n"
                + " IMAGE   - jpg png tif bmp webp \n\n"
                + " IMAGE LIST   - jpg png tif bmp webp \n\n"
                + " VIDEO   - mp4 webm mkv flv gif avi mov mpg qt 3gp \n\n")

    drag_drop = ttk.Notebook(root,
                            ondrop  = function_drop)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.75)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.75)/2

    drag_drop.place(x = x_center, 
                    y = y_center, 
                    width  = drag_drop_width * 0.75, 
                    height = drag_drop_height * 0.75)

    drag_drop_text = ttk.Label(root,
                            text    = text_drop,
                            ondrop  = function_drop,
                            font    = ft,
                            anchor  = "center",
                            relief  = 'flat',
                            justify = "center",
                            foreground = drag_drop_text_color)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.5)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.5)/2
    
    drag_drop_text.place(x = x_center, 
                    y = y_center, 
                    width  = drag_drop_width * 0.50, 
                    height = drag_drop_height * 0.50)

def show_video_info_with_drag_drop(video_path):
    global image
    
    fist_frame = "temp.png"
    
    clear_app_background()

    # 1) get video informations
    cap          = cv2.VideoCapture(video_path)
    width        = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    duration     = num_frames/frame_rate
    minutes      = int(duration/60)
    seconds      = duration % 60
    path_as_list = video_path.split("/")
    video_name   = str(path_as_list[-1])
    
    # 2) get first frame of the video
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(fist_frame, frame)
        break
    cap.release()

    # 3) resize the frame to fit the UI
    image_to_show_resized = resize_image_to_show(fist_frame)

    # 4) show the resized image in the UI
    image = tk.PhotoImage(file = image_to_show_resized)
    drag_drop_and_images = ttk.Label(root,
                                     image   = image,
                                     ondrop  = function_drop,
                                     anchor  = "center",
                                     relief  = "flat",
                                     justify = "center",
                                     background = background_color,
                                     foreground = "#202020")
    drag_drop_and_images.place(x = 30 + left_bar_width + drag_drop_width/2 - show_image_width/2,
                               y = drag_drop_height/2 - show_image_height/2 - 15,
                               width  = show_image_width,
                               height = show_image_height)

    # 5) remove the temp first frame
    os.remove(fist_frame)

    # 6) create string video description
    file_description = ( video_name + "\n"
                        + "[" + str(width) + "x" + str(height) + "]" + " | " + str(minutes) + 'm:' + str(round(seconds)) + "s | " + str(num_frames) + "frames | " + str(round(frame_rate)) + "fps")

    video_info_width = drag_drop_width * 0.8

    video_info_space = ttk.Label(root,
                                 text    = file_description,
                                 ondrop  = function_drop,
                                 font    = (default_font, round(11 * font_scale), "bold"),
                                 anchor  = "center",
                                 relief  = "flat",
                                 justify = "center",
                                 background = "#181818",
                                 foreground = "#D3D3D3",
                                 wraplength = video_info_width * 0.95)
    video_info_space.place(x = 30 + left_bar_width + drag_drop_width/2 - video_info_width/2,
                           y = drag_drop_height - 100,
                           width  = video_info_width,
                           height = 65)

    # 7) show clear button
    place_clean_button()

def show_list_images_in_GUI(image_list_prepared):
    clear_app_background()
    final_string = "\n"
    counter_img = 0

    for elem in image_list_prepared:
        counter_img += 1
        if counter_img <= 8:
            # add first 8 files in list
            img     = cv2.imread(elem.strip())
            width   = round(img.shape[1])
            height  = round(img.shape[0])
            img_name = str(elem.split("/")[-1])

            final_string += (str(counter_img) 
                            + ".  " 
                            + img_name 
                            + " | [" + str(width) + "x" + str(height) + "]" + "\n\n")
        else:
            final_string += "and others... \n"
            break

    list_height = 420
    list_width  = 750

    list_header = ttk.Label(root,
                            text    = " ❏  Image list ",
                            ondrop  = function_drop,
                            font    = (default_font, round(12 * font_scale), "bold"),
                            anchor  = "center",
                            relief  = "flat",
                            justify = "center",
                            background = "#181818",
                            foreground = "#D3D3D3")
    list_header.place(x = 30 + left_bar_width + drag_drop_width/2 - list_width/2,
                      y = drag_drop_height/2 - list_height/2 - 45,
                      width  = 200,
                      height = 36)

    multiple_images_list = ttk.Label(root,
                                     text=final_string,
                                     ondrop=function_drop,
                                     font=(default_font, round(11 * font_scale)),
                                     anchor="n",
                                     relief="flat",
                                     justify="left",
                                     background="#181818",
                                     foreground="#D3D3D3",
                                     wraplength=list_width - 10)
    multiple_images_list.place(x = 30 + left_bar_width + drag_drop_width/2 - list_width/2,
                               y = drag_drop_height/2 - list_height/2,
                               width  = list_width,
                               height = list_height)

    # then image counter
    multiple_images_label = ttk.Label(root,
                                      text    = str(len(image_list_prepared)) + ' images',
                                      ondrop  = function_drop,
                                      font    = (default_font, round(12 * font_scale), "bold"),
                                      anchor  = "center",
                                      relief  = "flat",
                                      justify = "center",
                                      background = "#181818",
                                      foreground = "#D3D3D3")
    multiple_images_label.place(x = 30 + left_bar_width + drag_drop_width/2 + 175,
                                y = drag_drop_height/2 + 220,
                                width  = 200,
                                height = 36)

def show_image_in_GUI(image_to_show, _ ):
    global image

    image_to_show = image_to_show.replace('{', '').replace('}', '')

    # 1) resize image to fit the UI
    image_to_show_resized = resize_image_to_show(image_to_show)

    # 2) clean the background
    clear_app_background()

    # 3) show the resized image in the UI
    image = tk.PhotoImage(file = image_to_show_resized)
    drag_drop_and_images = ttk.Label(root,
                                     text="",
                                     image   = image,
                                     ondrop  = function_drop,
                                     anchor  = "center",
                                     relief  = "flat",
                                     justify = "center",
                                     background = background_color,
                                     foreground = "#202020")
    drag_drop_and_images.place(x = 30 + left_bar_width + drag_drop_width/2 - show_image_width/2,
                               y = drag_drop_height/2 - show_image_height/2,
                               width  = show_image_width,
                               height = show_image_height)

    # 4) show the image file information in the UI
    path_as_list = image_to_show.split("/")
    img_name     = str(path_as_list[-1])
    img          = cv2.imread(image_to_show)
    width        = round(img.shape[1])
    height       = round(img.shape[0])

    single_image_path = (img_name
                         + " | [" + str(width) + "x" + str(height) + "]")
    single_image_info = ttk.Label(root,
                                  font=(default_font, round(10 * font_scale), "bold"),
                                  text = single_image_path,
                                  relief="flat",
                                  justify="center",
                                  background="#181818",
                                  foreground="#D3D3D3",
                                  anchor="center")

    single_image_info.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - image_text_height - 35,
                            width  = image_text_width,
                            height = image_text_height + 5)

    # 5) delete the resized temp image
    if "temp.png" in image_to_show_resized:
        os.remove("temp.png")

    # 6) show clear button
    place_clean_button()

def place_upscale_button():
    ft = tkFont.Font(family = default_font,
                    size   = round(11 * font_scale),
                    weight = 'bold')
    
    global play_icon
    play_icon = tk.PhotoImage(file = find_file_by_relative_path('upscale_icon.png'))
    
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = ft)

    Upscale_button = ttk.Button(root, 
                                text  = '  UPSCALE',
                                image = play_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Upscale_button.place(x      = 40 + left_bar_width/2 - 310/2,  
                         y      = left_bar_height - 100,
                         width  = 310,
                         height = 45)
    Upscale_button["command"] = lambda: upscale_button_command()

def place_stop_button():
    ft = tkFont.Font(family = default_font,
                    size   = round(11 * font_scale),
                    weight = 'bold')
    
    global stop_icon
    stop_icon = tk.PhotoImage(file = find_file_by_relative_path('stop_icon.png'))
    
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = ft)

    Stop_button = ttk.Button(root, 
                                text  = '  STOP UPSCALE ',
                                image = stop_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Stop_button.place(x      = 40 + left_bar_width/2 - 310/2,  
                      y      = left_bar_height - 100,
                      width  = 310,
                      height = 45)

    Stop_button["command"] = lambda: stop_button_command()

def combobox_AI_selection(event):
    global AI_model

    selected = str(selected_AI.get())

    if 'Fused' in selected:
        AI_model = 'Fused'
    else:
        AI_model = selected

    Combo_box_AI.set('')
    Combo_box_AI.set(selected)

def combobox_upscale_factor_selection(event):
    global upscale_factor

    selected = str(selected_upscale_factor.get())
    if '1' in selected:
        upscale_factor = 1
    elif '2' in selected:
        upscale_factor = 2
    elif '3' in selected:
        upscale_factor = 3
    elif '4' in selected:
        upscale_factor = 4

    Combo_box_upscale_factor.set('') # clean selection in widget
    Combo_box_upscale_factor.set(selected)

def combobox_backend_selection(event):
    global device

    selected = str(selected_backend.get())
    if 'gpu' in selected:
        device = "dml"
    elif 'cpu' in selected:
        device = "cpu"

    Combo_box_backend.set('')
    Combo_box_backend.set(selected)

def combobox_VRAM_selection(event):
    global tiles_resolution

    selected = str(selected_VRAM.get())

    if '2' in selected:
        tiles_resolution = 300
    if '3' in selected:
        tiles_resolution = 400
    if '4' in selected:
        tiles_resolution = 500
    if '5' in selected:
        tiles_resolution = 600
    if '6' in selected:
        tiles_resolution = 700
    if '8' in selected:
        tiles_resolution = 800
    if '12' in selected:
        tiles_resolution = 1000

    Combo_box_VRAM.set('')
    Combo_box_VRAM.set(selected)

def place_backend_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    sv_ttk.use_dark_theme()

    root.option_add("*TCombobox*Listbox*Background", background_color)
    root.option_add("*TCombobox*Listbox*Foreground", selected_button_color)
    root.option_add("*TCombobox*Listbox*Font",       ft)
    root.option_add('*TCombobox*Listbox.Justify',    'center')

    global Combo_box_backend
    Combo_box_backend = ttk.Combobox(root, 
                            textvariable = selected_backend, 
                            justify      = 'center',
                            foreground   = '#F5F5F5',
                            values       = ['GPU', 'CPU'],
                            state        = 'readonly',
                            takefocus    = False,
                            font         = ft)
    Combo_box_backend.place(x = 40 + left_bar_width/2 - 285/2, 
                       y = button_3_y, 
                       width  = 285, 
                       height = 42)
    Combo_box_backend.bind('<<ComboboxSelected>>', combobox_backend_selection)
    Combo_box_backend.set('GPU')

def place_upscale_factor_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    sv_ttk.use_dark_theme()  # Set SunValley dark theme

    root.option_add("*TCombobox*Listbox*Background", background_color)
    root.option_add("*TCombobox*Listbox*Foreground", selected_button_color)
    root.option_add("*TCombobox*Listbox*Font",       ft)
    root.option_add('*TCombobox*Listbox.Justify', 'center')

    global Combo_box_upscale_factor
    Combo_box_upscale_factor = ttk.Combobox(root, 
                            textvariable = selected_upscale_factor, 
                            justify      = 'center',
                            foreground   = '#F5F5F5',
                            values       = ['x1', 'x2', 'x3', 'x4'],
                            state        = 'readonly',
                            takefocus    = False,
                            font         = ft)
    Combo_box_upscale_factor.place(x = 40 + left_bar_width/2 - 285/2, 
                       y = button_2_y, 
                       width  = 285, 
                       height = 42)
    Combo_box_upscale_factor.bind('<<ComboboxSelected>>', combobox_upscale_factor_selection)
    Combo_box_upscale_factor.set('x2')

def place_AI_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    sv_ttk.use_dark_theme()  # Set SunValley dark theme

    root.option_add("*TCombobox*Listbox*Background", background_color)
    root.option_add("*TCombobox*Listbox*Foreground", selected_button_color)
    root.option_add("*TCombobox*Listbox*Font",       ft)
    root.option_add('*TCombobox*Listbox.Justify', 'center')

    global Combo_box_AI
    Combo_box_AI = ttk.Combobox(root, 
                        textvariable = selected_AI, 
                        justify      = 'center',
                        foreground   = '#F5F5F5',
                        values       = ['BSRGAN', 'RealSR_JPEG', 'Fused [BSRGAN + RealSR_JPEG]'],
                        state        = 'readonly',
                        takefocus    = False,
                        font         = ft)
    Combo_box_AI.place(x = 40 + left_bar_width/2 - 285/2,  
                       y = button_1_y, 
                       width  = 285, 
                       height = 42)
    Combo_box_AI.bind('<<ComboboxSelected>>', combobox_AI_selection)
    Combo_box_AI.set(AI_model)

def place_VRAM_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    sv_ttk.use_dark_theme()

    root.option_add("*TCombobox*Listbox*Background", background_color)
    root.option_add("*TCombobox*Listbox*Foreground", selected_button_color)
    root.option_add("*TCombobox*Listbox*Font",       ft)
    root.option_add('*TCombobox*Listbox.Justify',    'center')

    global Combo_box_VRAM
    Combo_box_VRAM = ttk.Combobox(root, 
                            textvariable = selected_VRAM, 
                            justify      = 'center',
                            foreground   = '#F5F5F5',
                            values       = ['2Gb', '3Gb', '4Gb', '5Gb', '6Gb', '8Gb', '12Gb'],
                            state        = 'readonly',
                            takefocus    = False,
                            font         = ft)
    Combo_box_VRAM.place(x = 40 + left_bar_width/2 - 285/2, 
                         y = button_4_y, 
                         width  = 285, 
                         height = 42)
    Combo_box_VRAM.bind('<<ComboboxSelected>>', combobox_VRAM_selection)
    Combo_box_VRAM.set('4Gb')

def place_clean_button():
    ft = tkFont.Font(family = default_font,
                    size   = round(11 * font_scale),
                    weight = 'bold')
    
    global clear_icon
    clear_icon = tk.PhotoImage(file = find_file_by_relative_path('clear_icon.png'))
    
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = ft)

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = tk.LEFT,
                            style    = 'Bold.TButton')

    clean_button.place(x = 30 + left_bar_width + drag_drop_width/2 - 260/2,
                       y = 15,
                       width  = 260,
                       height = 45)
    clean_button["command"] = lambda: place_drag_drop_widget()

def place_title():
    Title = ttk.Label(root, 
                      font = (default_font, round(16 * font_scale), "bold"),
                      foreground = "#DA70D6", 
                      anchor     = 'center', 
                      text       = "QualityScaler")
    Title.place(x = 55,
                y = 25,
                width  = left_bar_width/2,
                height = 55)

def place_blurred_background():
    Background = ttk.Label(root, background = background_color, relief = 'flat')
    Background.place(x = 0, 
                     y = 0, 
                     width  = window_width,
                     height = window_height)

def place_left_bar():
    Left_bar = ttk.LabelFrame(root)
    Left_bar.place(x = 40, 
                   y = -12, 
                   width  = left_bar_width,
                   height = 710)

def place_github_button():
    global logo_git
    logo_git = PhotoImage(file = find_file_by_relative_path("github_logo.png"))
    logo_git_label                     = tk.Button(root)
    logo_git_label['image']            = logo_git
    logo_git_label["justify"]          = "center"
    logo_git_label["bg"]               = "grey"
    logo_git_label["relief"]           = "flat"
    logo_git_label["activebackground"] = "grey"
    logo_git_label["borderwidth"]      = 1
    logo_git_label.place(x = left_bar_width - 140,
                            y = 35,
                            width  = 37,
                            height = 37)
    logo_git_label["command"] = lambda: opengithub()

def place_paypal_button():
    global logo_paypal
    logo_paypal = PhotoImage(file=find_file_by_relative_path("paypal_logo.png"))
    logo_paypal_label                     = tk.Button(root)
    logo_paypal_label['image']            = logo_paypal
    logo_paypal_label["justify"]          = "center"
    logo_paypal_label["bg"]               = "black"
    logo_paypal_label["relief"]           = "flat"
    logo_paypal_label["activebackground"] = "grey"
    logo_paypal_label["borderwidth"]      = 1
    logo_paypal_label.place(x = left_bar_width - 90,
                            y = 35,
                            width  = 37,
                            height = 37)
    logo_paypal_label["command"] = lambda: openpaypal()

def place_patreon_button():
    global logo_patreon
    logo_patreon = PhotoImage(file=find_file_by_relative_path("patreon_logo.png"))
    logo_patreon_label            = tk.Button(root)
    logo_patreon_label['image']   = logo_patreon
    logo_patreon_label["justify"] = "center"
    logo_patreon_label["bg"]      = "black"
    logo_patreon_label["relief"]  = "flat"
    logo_patreon_label["activebackground"] = "grey"
    logo_patreon_label["borderwidth"]      = 1
    logo_patreon_label.place(x = left_bar_width - 40,
                                y = 35,
                                width  = 37,
                                height = 37)
    logo_patreon_label["command"] = lambda: openpatreon()

def place_AI_models_title():
    IA_selection_title = ttk.Label(root, 
                                   font = (default_font, round(12 * font_scale), "bold"), 
                                   foreground = '#F5F5F5', 
                                   justify    = 'center', 
                                   relief     = 'flat', 
                                   text       = " ◪  AI model ")
    IA_selection_title.place(x = left_bar_width/2 - 174 + 45,
                             y = button_1_y - 45,
                             width  = 348,
                             height = 40)

def place_upscale_factor_title():
    Upscale_fact_selection_title = ttk.Label(root, 
                                            font = (default_font, round(12 * font_scale), "bold"), 
                                            foreground = '#F5F5F5', 
                                            justify    = 'center', 
                                            relief     = 'flat', 
                                            text       = " ⤮  Upscale factor ")
    Upscale_fact_selection_title.place(x = left_bar_width/2 - 175 + 45,
                                        y = button_2_y - 45,
                                        width  = 155,
                                        height = 40)

def place_backend_title():
    Upscale_backend_selection_title = ttk.Label(root, 
                                                font = (default_font, round(12 * font_scale), "bold"), 
                                                foreground = '#F5F5F5', 
                                                justify    = 'center', 
                                        	    relief     = 'flat', 
                                                text       = " ⍚  AI backend ")
    Upscale_backend_selection_title.place(x = left_bar_width/2 - 175 + 45,
                                          y = button_3_y - 45,
                                          width  = 145,
                                          height = 40)

def place_VRAM_title():
    IA_selection_title = ttk.Label(root, 
                                   font = (default_font, round(12 * font_scale), "bold"), 
                                   foreground = '#F5F5F5', 
                                   justify    = 'center', 
                                   relief     = 'flat', 
                                   text       = " ⋈  Vram/Ram to use ")
    IA_selection_title.place(x = left_bar_width/2 - 174 + 45,
                             y = button_4_y - 45,
                             width  = 348,
                             height = 40)

def place_message_box():
    message_label = ttk.Label(root,
                            font = (default_font, round(11 * font_scale), "bold"),
                            textvar    = info_string,
                            relief     = "flat",
                            justify    = "center",
                            foreground = "#ffbf00",
                            anchor     = "center")
    message_label.place(x = 40 + left_bar_width/2 - (left_bar_width * 0.75)/2,
                        y = 585,
                        width  = left_bar_width * 0.75,
                        height = 30)

# ---------------------- /GUI related ----------------------

# ---------------------- /Functions ----------------------

def apply_windows_dark_bar(window_root):
    window_root.update()
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    set_window_attribute          = ctypes.windll.dwmapi.DwmSetWindowAttribute
    get_parent                    = ctypes.windll.user32.GetParent
    hwnd                          = get_parent(window_root.winfo_id())
    rendering_policy              = DWMWA_USE_IMMERSIVE_DARK_MODE
    value                         = ctypes.c_int(2)
    set_window_attribute(hwnd, rendering_policy, ctypes.byref(value), ctypes.sizeof(value))    
    window_root.update()

def apply_windows_transparency_effect(window_root):
    window_root.wm_attributes("-transparent", background_color)
    hwnd = ctypes.windll.user32.GetParent(window_root.winfo_id())
    ApplyMica(hwnd, MICAMODE.DARK )

class ErrorMessage():
    def __init__(self, error_root, error_type):
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
        scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

        if scaleFactor == 1.0:
            font_scale = 1.2
        elif scaleFactor == 1.25:
            font_scale = 1.0
        else:
            font_scale = 0.8

        error_root.title("")
        width  = 515
        height = 525
        screenwidth = error_root.winfo_screenwidth()
        screenheight = error_root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        error_root.geometry(alignstr)
        error_root.resizable(width=False, height=False)

        error_root.iconbitmap(create_void_logo())

        window_width  = 515
        window_height = 530

        error_string  = "Upscale\nerror"
        error_suggest = (" Ops, some error occured while upscaling: \n\n"
                             + " - have you changed the file location? \n"
                             + " - try to set Upscale Factor to x2 or x3 \n"
                             + " - try to set AI Backend to <cpu> ")

        ft = tkFont.Font(family=default_font,
                         size=int(14 * font_scale),
                         weight="bold")

        Error_container = tk.Label(error_root)
        Error_container["anchor"]  = "center"
        Error_container["justify"] = "center"
        Error_container["font"]    = ft
        Error_container["bg"]      = "#FF4433"
        Error_container["fg"]      = "#202020"
        Error_container["text"]    = error_string
        Error_container["relief"]  = "flat"
        Error_container.place(x = 0,
                              y = 0,
                              width  = window_width,
                              height = window_height/4)

        ft = tkFont.Font(family=default_font,
                    size=int(13 * font_scale),
                    weight="bold")

        Suggest_container = tk.Label(error_root)
        Suggest_container["anchor"]  = "center"
        Suggest_container["justify"] = "left"
        Suggest_container["font"]    = ft
        Suggest_container["bg"]      = background_color
        Suggest_container["fg"]      = "grey"
        Suggest_container["text"]    = error_suggest
        Suggest_container["relief"]  = "flat"
        Suggest_container["wraplength"] = window_width*0.9
        Suggest_container.place(x = 0,
                                y = window_height/4,
                                width  = window_width,
                                height = window_height*0.75)

        error_root.attributes('-topmost', True)
        
        apply_windows_dark_bar(error_root)
        apply_windows_transparency_effect(error_root)

        error_root.update()
        error_root.mainloop()

class App:
    def __init__(self, root):
        root.title('')
        width        = window_width
        height       = window_height
        screenwidth  = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr     = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        root.iconphoto(False, PhotoImage(file = find_file_by_relative_path("logo.png")))

        place_blurred_background()       # Background
        place_left_bar()                 # Left container
        place_title()                    # Qualityscaler title

        place_github_button()            # Github
        place_paypal_button()            # Paypal
        place_patreon_button()           # Patreon

        place_AI_models_title()          # AI models title
        place_AI_combobox()              # AI models widget

        place_upscale_factor_title()     # Upscale factor title
        place_upscale_factor_combobox()  # Upscale factor widget

        place_backend_title()            # Backend title
        place_backend_combobox()         # Backend widget

        place_VRAM_title()               # VRAM title
        place_VRAM_combobox()

        place_message_box()              # Message box

        place_upscale_button()           # Upscale button

        place_drag_drop_widget()         # Drag&Drop widget
        
        apply_windows_transparency_effect(root)
        apply_windows_dark_bar(root)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    root        = tkinterDnD.Tk()
    info_string = tk.StringVar()
    selected_AI = tk.StringVar()
    selected_upscale_factor = tk.StringVar()
    selected_backend = tk.StringVar()
    selected_VRAM    = tk.StringVar()

    app = App(root)
    root.update()
    root.mainloop()
